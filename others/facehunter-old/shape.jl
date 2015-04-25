
using Images
using ImageView
using Color
using FixedPointNumbers

function nonzero_indexes{T <: Number}(mat::Matrix{T})
    idxs = (Int, Int)[]
    for i=1:size(mat, 1)
        for j=1:size(mat, 2)
            if mat[i, j] != 0
                push!(idxs, (i, j))
            end
        end
    end
    return idxs
end


function collect_nonzeros{T <: Number}(mat::Matrix{T}, idxs::Array{(Int, Int)})
    arr = zeros(T, length(idxs))
    k = 1
    for (i, j) in idxs
        arr[k] = mat[i, j]
        k += 1
    end
    return arr
end


function map_nonzeros{T <: Number}(matsize::(Int, Int), arr::Array{T, 1},
                      idxs::Array{(Int, Int)})
    mat = zeros(T, matsize)
    k = 1
    for (i, j) in idxs
        mat[i, j] = arr[k]
        k += 1
    end
    return mat
end
