#include <list>
#include <mutex>
#include <atomic>
#include <thread>
#include <vector>
#include <cassert>
#include <condition_variable>

#include <stdint.h>
#include <string.h>
#include <pthread.h>
#include <immintrin.h>

#include <faiss/impl/IVFFlatx.h>

namespace faiss {

//=============================Concurrent context=============================

class Context {

public:
    struct Task {
        const void* x;
        size_t iquery;
        size_t ilist;
        size_t iy_start;
        size_t iy_end;
    };

private:
    const IVFFlatx* ivfflatx;
    const Task** tasks;
    size_t ntask;
    std::atomic<size_t> cursor;
    void (*func)(size_t, size_t, size_t, float, void*);
    void* arg;

public:
    Context(const IVFFlatx* ivfflatx, const Task** tasks, size_t ntask,
            void (*func)(size_t, size_t, size_t, float, void*), void* arg):
            ivfflatx(ivfflatx), tasks(tasks), ntask(ntask), cursor(0),
            func(func), arg(arg) {
    }

    void work() {
        while (true) {
            size_t itask = cursor++;
            if (itask >= ntask) {
                break;
            }
            const Task* task = tasks[itask];
            const void* x = task->x;
            size_t iquery = task->iquery;
            size_t ilist = task->ilist;
            size_t iy_start = task->iy_start;
            size_t iy_end = task->iy_end;
            for (size_t iy = iy_start; iy < iy_end; iy++) {
                float dis = ivfflatx->get_distance(ilist, iy, x);
                func(iquery, ilist, iy, dis, arg);
            }
        }
    }
};

//================================Thread pool=================================

class Thread {

private:
    std::mutex mutex;
    std::condition_variable condv;
    std::atomic<Context*> context;
    std::thread thread;

private:
    Thread(): context(nullptr), thread([&]() {
                routine();
            }) {
    }

    ~Thread() {
        std::unique_lock<std::mutex> lock(mutex);
        context.store((Context*)(-1));
        condv.notify_one();
        lock.unlock();
        thread.join();
    }

    void routine() {
        while (true) {
            std::unique_lock<std::mutex> lock(mutex);
            Context* c = context.load();
            if (!c) {
                condv.wait(lock);
            }
            c = context.load();
            assert(c);
            if (c == (const Context*)(-1)) {
                break;
            }
            c->work();
            context.store(nullptr);
        }
    }

public:
    void assign(Context* c) {
        assert(c);
        std::unique_lock<std::mutex> lock(mutex);
        assert(!context.load());
        context.store(c);
        condv.notify_one();
    }

    void join() {
        while (context.load());
    }

private:
    struct FreeList {
        std::list<Thread*> list;
        pthread_spinlock_t lock;

        FreeList() {
            pthread_spin_init(&lock, 0);
        }

        ~FreeList() {
            pthread_spin_destroy(&lock);
        }

        Thread* pop() {
            Thread* thread;
            pthread_spin_lock(&lock);
            if (list.empty()) {
                thread = nullptr;
            }
            else {
                thread = list.back();
                list.pop_back();
            }
            pthread_spin_unlock(&lock);
            return thread;
        }

        void push(Thread* thread) {
            pthread_spin_lock(&lock);
            list.push_back(thread);
            pthread_spin_unlock(&lock);
        }

        void push(Thread** threads, size_t count) {
            pthread_spin_lock(&lock);
            list.insert(list.end(), threads, threads + count);
            pthread_spin_unlock(&lock);
        }
    };

    static const size_t nlist = 8;
    static FreeList lists[nlist];

private:
    static inline size_t select_list() {
        static const size_t shift = 10; 
        uint8_t stack_position;
        return (((size_t)&stack_position) >> shift) % nlist;
    }

public:
    static Thread* allocate() {
        Thread* thread;
        size_t barrier = select_list();
        for (size_t i = barrier; i < nlist; i++) {
            thread = lists[i].pop();
            if (thread) {
                return thread;
            }
        }
        for (size_t i = 0; i < barrier; i++) {
            thread = lists[i].pop();
            if (thread) {
                return thread;
            }
        }
        return new Thread;
    }

    static void release(Thread* thread) {
        lists[select_list()].push(thread);
    }

    static void release(Thread** threads, size_t count) {
        lists[select_list()].push(threads, count);
    }
};

Thread::FreeList Thread::lists[Thread::nlist];

//========================IVFFlatx generic implementation=====================

IVFFlatx::IVFFlatx(size_t dim, size_t nlist): dim(dim), nlist(nlist),
        ntotal(0) {
    assert(dim > 0);
    assert(nlist > 0);
}

IVFFlatx::~IVFFlatx() {
}

float IVFFlatx::get_distance(size_t ilist, size_t iy, const float* x) const {
    const void* converted_x = convert_vector(x);
    float dis = get_distance(ilist, iy, converted_x);
    del_converted_vector(converted_x);
    return dis;
}

void IVFFlatx::traverse(size_t nquery, const IVFFlatx::Query* queries,
        void (*func)(size_t, size_t, size_t, float, void*),
        void* arg,
        size_t thread_count, size_t batch_size) const {
    assert(thread_count > 0);
    assert(batch_size > 0);
    if (thread_count == 1) {
        for (size_t iquery = 0; iquery < nquery; iquery++) {
            const IVFFlatx::Query* query = queries + iquery;
            const void* x = convert_vector(query->x);
            size_t nprobe = query->nprobe;
            const size_t* ilists = query->ilists;
            for (size_t iprobe = 0; iprobe < nprobe; iprobe++) {
                size_t ilist = ilists[iprobe];
                size_t ny = get_list_size(ilist);
                for (size_t iy = 0; iy < ny; iy++) {
                    float dis = get_distance(ilist, iy, x);
                    func(iquery, ilist, iy, dis, arg);
                }
            }
            del_converted_vector(x);
        }
        return;
    }
    std::vector<const Context::Task*> tasks;
    for (size_t iquery = 0; iquery < nquery; iquery++) {
        const IVFFlatx::Query* query = queries + iquery;
        const void* x = convert_vector(query->x);
        size_t nprobe = query->nprobe;
        const size_t* ilists = query->ilists;
        for (size_t iprobe = 0; iprobe < nprobe; iprobe++) {
            size_t ilist = ilists[iprobe];
            size_t ny = get_list_size(ilist);
            for (size_t iy = 0; iy < ny; iy += batch_size) {
                Context::Task* task = new Context::Task {
                    .x = x,
                    .iquery = iquery,
                    .ilist = ilist,
                    .iy_start = iy,
                    .iy_end = std::min(iy + batch_size, ny),
                };
                tasks.push_back(task);
            }
        }
    }
    size_t ntask = tasks.size();
    Context context(this, tasks.data(), ntask, func, arg);
    thread_count--;
    Thread* threads[thread_count];
    for (size_t i = 0; i < thread_count; i++) {
        Thread* thread = Thread::allocate();
        threads[i] = thread;
        thread->assign(&context);
    }
    context.work();
    for (size_t i = 0; i < thread_count; i++) {
        threads[i]->join();
    }
    Thread::release(threads, thread_count);
    size_t last_iquery = (size_t)(-1);
    for (size_t i = 0; i < ntask; i++) {
        const Context::Task* task = tasks[i];
        if (task->iquery != last_iquery) {
            del_converted_vector(task->x);
            last_iquery = task->iquery;
        }
        delete task;
    }
    assert(last_iquery + 1 == nquery);
}

//=====================Generic and optimized Convertion=======================

template <typename T>
const T* cast_vector(size_t dim, const float* x, T*) {
    T* newx = new T[dim];
    for (size_t i = 0; i < dim; i++) {
        newx[i] = static_cast<T>(x[i]);
    }
    return newx;
}

template <typename T>
inline void del_casted_vector(const T* converted_x) {
    delete[] converted_x;
}

inline const float* cast_vector(size_t /*dim*/, const float* x, float*) {
    return x;
}

inline void del_casted_vector(const float* /*converted_x*/) {
}

//===========================inner product functions==========================

template <typename Tx, typename Ty>
float get_inner_product(size_t dim, const Tx* x, const Ty* y) {
    assert(false);
    float ip = 0;
    for (size_t i = 0; i < dim; i++) {
        ip += x[i] * y[i];
    }
    return ip;
}

float get_inner_product(size_t dim, const float* x, const float* y) {
    __m128 msum = _mm_setzero_ps();
    while (dim >= 4) {
        __m128 mx = _mm_load_ps(x);
        __m128 my = _mm_load_ps(y);
        msum = _mm_add_ps(msum, _mm_mul_ps(mx, my));
        x += 4;
        y += 4;
        dim -= 4;
    }
    msum = _mm_hadd_ps (msum, msum);
    msum = _mm_hadd_ps (msum, msum);
    float ip = _mm_cvtss_f32(msum);
    if (dim > 0) {
        for (size_t i = 0; i < dim; i++) {
            ip += x[i] * y[i];
        }
    }
    return ip;
}

float get_inner_product(size_t dim, const int32_t* x, const int32_t* y) {
    __m128i msum = _mm_setzero_si128();
    while (dim >= 4) {
        __m128i mx = _mm_load_si128((const __m128i*)x);
        __m128i my = _mm_load_si128((const __m128i*)y);
        msum = _mm_add_epi64(msum, _mm_mul_epi32(mx, my));
        mx = _mm_shuffle_epi32(mx, _MM_SHUFFLE(0, 3, 0, 1));
        my = _mm_shuffle_epi32(my, _MM_SHUFFLE(0, 3, 0, 1));
        msum = _mm_add_epi64(msum, _mm_mul_epi32(mx, my));
        x += 4;
        y += 4;
        dim -= 4;
    }
    const int64_t* sum = (const int64_t*)&msum;
    int64_t ip = sum[0] + sum[1];
    if (dim > 0) {
        for (size_t i = 0; i < dim; i++) {
            ip += x[i] * y[i];
        }
    }
    return ip;
}

float get_inner_product(size_t dim, const int16_t* x, const int16_t* y) {
    static const __m128i mzero = _mm_setzero_si128();
    __m128i msum = _mm_setzero_si128();
    while (dim >= 4) {
        __m128i mx = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)x),
                mzero);
        __m128i my = _mm_unpacklo_epi16(_mm_loadl_epi64((const __m128i*)y),
                mzero);
        msum = _mm_add_epi32(msum, _mm_mullo_epi32(mx, my));
        x += 4;
        y += 4;
        dim -= 4;
    }
    msum = _mm_hadd_epi32(msum, msum);
    msum = _mm_hadd_epi32(msum, msum);
    int32_t ip = msum[0];
    if (dim > 0) {
        for (size_t i = 0; i < dim; i++) {
            ip += x[i] * y[i];
        }
    }
    return ip;
}

float get_inner_product(size_t dim, const int8_t* x, const int8_t* y) {
    static const __m128i mzero = _mm_setzero_si128();
    __m128i msum = _mm_setzero_si128();
    while (dim >= 8) {
        __m128i mx = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)x),
                mzero);
        __m128i my = _mm_unpacklo_epi8(_mm_loadl_epi64((const __m128i*)y),
                mzero);
        msum = _mm_adds_epi16(msum, _mm_mullo_epi16(mx, my));
        x += 8;
        y += 8;
        dim -= 8;
    }
    msum = _mm_hadds_epi16(msum, msum);
    msum = _mm_hadds_epi16(msum, msum);
    msum = _mm_hadds_epi16(msum, msum);
    const int16_t* sum = (const int16_t*)&msum;
    int16_t ip = sum[0];
    if (dim > 0) {
        for (size_t i = 0; i < dim; i++) {
            ip += x[i] * y[i];
        }
    }
    return ip;
}

//==============================Byte based kernels============================

template <typename T>
class ByteBasedIP : public IVFFlatx {

private:
    struct List {
        std::vector<T> raw;
        size_t count;

        List(): count(0) {
        }
    };

    List* lists;

public:
    ByteBasedIP(size_t dim, size_t nlist): IVFFlatx(dim, nlist) {
        lists = new List[nlist];
    }

    virtual ~ByteBasedIP() {
        delete[] lists;
    }

    virtual size_t add(size_t ilist, const float* y) override {
        assert(ilist < nlist);
        List* list = lists + ilist;
        for (size_t i = 0; i < dim; i++) {
            list->raw.push_back(static_cast<T>(y[i]));
        }
        size_t iy = list->count++;
        ntotal++;
        return iy;
    }

    virtual size_t get_list_size(size_t ilist) const override {
        assert(ilist < nlist);
        return lists[ilist].count;
    }

    virtual const void* convert_vector(const float* x) const override {
        return cast_vector(dim, x, (T*)nullptr);
    }

    virtual void del_converted_vector(const void* converted_x)
            const override {
        del_casted_vector((const T*)converted_x);
    }

    virtual float get_distance(size_t ilist, size_t iy,
            const void* converted_x) const override {
        assert(ilist < nlist);
        const T* y = lists[ilist].raw.data() + iy * dim;
        return get_inner_product(dim, (const T*)converted_x, y);
    }
};

template <typename T>
class ByteBasedL2 : public ByteBasedIP<T> {

private:
    struct ConvertedX {
        const T* x;
        float norm_x;
    };

    std::vector<float>* norms;

public:
    ByteBasedL2(size_t dim, size_t nlist): ByteBasedIP<T>(dim, nlist) {
        norms = new std::vector<float>[nlist];
    }

    virtual ~ByteBasedL2() {
        delete[] norms;
    }

    virtual size_t add(size_t ilist, const float* y) override {
        size_t iy = ByteBasedIP<T>::add(ilist, y);
        norms[ilist].push_back(get_inner_product(this->get_dim(), y, y));
        return iy;
    }

    virtual const void* convert_vector(const float* x) const override {
        ConvertedX* conv = new ConvertedX;
        conv->x = cast_vector(this->get_dim(), x, (T*)nullptr);
        conv->norm_x = get_inner_product(this->get_dim(), x, x);
        return conv;
    }

    virtual void del_converted_vector(const void* converted_x) const {
        const ConvertedX* conv = (const ConvertedX*)converted_x;
        del_casted_vector(conv->x);
        delete conv;
    }

    virtual float get_distance(size_t ilist, size_t iy,
            const void* converted_x) const override {
        const ConvertedX* conv = (const ConvertedX*)converted_x; 
        float ip = ByteBasedIP<T>::get_distance(ilist, iy, conv->x);
        return conv->norm_x + norms[ilist][iy] - 2 * ip;
    }
};

//============================Static Factory==================================

IVFFlatx* IVFFlatx::build(size_t dim, size_t nlist, const char* type) {
    if (strcasecmp(type, "FP32IP") == 0) {
        return new ByteBasedIP<float>(dim, nlist);
    }
    else if (strcasecmp(type, "FP32L2") == 0) {
        return new ByteBasedL2<float>(dim, nlist);
    }
    else if (strcasecmp(type, "INT32IP") == 0) {
        return new ByteBasedIP<int32_t>(dim, nlist);
    }
    else if (strcasecmp(type, "INT32L2") == 0) {
        return new ByteBasedL2<int32_t>(dim, nlist);
    }
    else if (strcasecmp(type, "INT16IP") == 0) {
        return new ByteBasedIP<int16_t>(dim, nlist);
    }
    else if (strcasecmp(type, "INT16L2") == 0) {
        return new ByteBasedL2<int16_t>(dim, nlist);
    }
    else if (strcasecmp(type, "INT8IP") == 0) {
        return new ByteBasedIP<int8_t>(dim, nlist);
    }
    else if (strcasecmp(type, "INT8L2") == 0) {
        return new ByteBasedL2<int8_t>(dim, nlist);
    }
    return nullptr;
}

}