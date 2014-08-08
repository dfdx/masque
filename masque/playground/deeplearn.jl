
using Images
using ImageView
using MiLK.NNet

include("interp.jl")


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


#### normalize image data to [0..1]!!!
#### add check to fit! method!!!

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


IMSIZE = (96, 96)

# Create dataset consiting of nonzero pixels of face images
# Returns:
#   dat : Matrix(# of nonzeros x # of filenames) - data matrix
#   nonzero_idxs : Vector((i, j)) - nonzero pixel coordinates
function facedata(datadir="../../data/CK/faces_aligned", imsize=IMSIZE)
    filenames = readdir(datadir)
    refmat = convert(Array, imread(datadir * "/" * filenames[1]))
    refmat = imresize(refmat, imsize)
    refmat = refmat ./ 256
    nonzero_idxs = nonzero_indexes(refmat)
    dat = zeros(length(nonzero_idxs), length(filenames))
    for (i, fname) in enumerate(filenames)
        println(i, " ", fname)
        mat = convert(Array, imread(datadir * "/" * fname))
        mat = imresize(mat, imsize)
        mat = mat ./ 256
        dat[:, i] = collect_nonzeros(mat, nonzero_idxs)
    end
    return dat, nonzero_idxs
end

# view weight
#  W - wright matrix
#  n - index of weight to view
#  nzs - nonzero pixel mapping
function vw(W, n, nzs)
    w = W[n, :]
    w = reshape(w, length(w))
    wim = map_nonzeros(IMSIZE, w, nzs)
    wim = wim ./ (maximum(wim) - minimum(wim))
    view(wim)
end


function wplot(W, nzs, padding=10)
    h, w = IMSIZE
    n = size(W, 1)
    rows = int(floor(sqrt(n)))
    cols = int(ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))
    for i=1:n
        w = W[i, :]
        w = reshape(w, length(w))
        wim = map_nonzeros(IMSIZE, w, nzs)
        wim = wim ./ (maximum(wim) - minimum(wim))
        r = div(i, rows) + 1
        c = rem(i, rows)
        # dat[(r * (h + padding) - 1 + halfpad) : (r * (h + padding) - halfpad),
        #     (c * (w + padding) - 1 + halfpad) : (c * (w + padding) - halfpad)] = wim
    end
    view(dat)
    return dat
end


function run1()
    dat, nzs = facedata()
    n_feat, n_samples = size(dat)
    model = RBM(n_feat, int(n_feat / 4))
    for i=1:3
        println("meta-iteration #", i)
        fit!(model, dat, n_iter=10)
        println("Sleeping for 20 seconds to cool down")
        sleep(20)
    end
    w = model.weights[1, :]
    w = reshape(w, length(w))
    wim = map_nonzeros(IMSIZE, w, nzs)
    wim = wim ./ (maximum(wim) - minimum(wim))
    view(wim)
    return model, nzs
end



# REPL

if isinteractive()
    im = imread("../../data/CK/faces_aligned/S005_001_00000010.png")
    mat = convert(Array, im)
end
