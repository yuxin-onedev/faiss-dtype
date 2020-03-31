#ifndef IVFFLATX_H
#define IVFFLATX_H

#include <stdlib.h>

namespace faiss {

class IVFFlatx {

    friend class Context;

protected:
    const size_t dim;
    const size_t nlist;
    size_t ntotal;

protected:
    IVFFlatx(size_t dim, size_t nlist);

    virtual const void* convert_vector(const float* x) const = 0;

    virtual void del_converted_vector(const void* converted_x) const = 0;

    virtual float get_distance(size_t ilist, size_t iy,
            const void* converted_x) const = 0;

public:
    virtual ~IVFFlatx();

    size_t inline get_dim() const {
        return dim;
    }

    size_t inline get_nlist() const {
        return nlist;
    }

    size_t inline get_ntotal() const {
        return ntotal;
    }

    virtual size_t add(size_t ilist, const float* y) = 0;

    virtual size_t get_list_size(size_t ilist) const = 0;

    virtual float get_distance(size_t ilist, size_t iy,
            const float* x) const;

    struct Query {
        const float* x;
        size_t nprobe;
        const size_t* ilists;
    };

    void traverse(size_t nquery, const Query* queries,
            void (*func)(size_t iquery, size_t ilist, size_t iy,
                    float distance, void* arg),
            void* arg,
            size_t thread_count = 1, size_t batch_size = 8) const;

public:
    static IVFFlatx* build(size_t dim, size_t nlist, const char* type);
};

}

#endif