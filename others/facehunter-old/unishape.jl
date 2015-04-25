
using Images
using ImageView
using Color
using FixedPointNumbers
using Boltzmann
using HDF5, JLD

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


IMSIZE = (96, 96)

function imresize!(resized, original)
    scale1 = (size(original,1)-1)/(size(resized,1)-0.999f0)
    scale2 = (size(original,2)-1)/(size(resized,2)-0.999f0)
    for jr = 0:size(resized,2)-1
        jo = scale2*jr
        ijo = itrunc(jo)
        fjo = jo - oftype(jo, ijo)
        @inbounds for ir = 0:size(resized,1)-1
            io = scale1*ir
            iio = itrunc(io)
            fio = io - oftype(io, iio)
            tmp = (1-fio)*((1-fjo)*original[iio+1,ijo+1] +
                           fjo*original[iio+1,ijo+2]) +
            fio*((1-fjo)*original[iio+2,ijo+1] +
                   fjo*original[iio+2,ijo+2])
            resized[ir+1,jr+1] = convert(eltype(resized), tmp)
        end
    end
    resized
end

imresize(original, new_size) =
    imresize!(similar(original,
                      new_size), original)
                      
convertsafely{T<:FloatingPoint}(::Type{T}, val) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::Integer) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::FloatingPoint) =
    itrunc(T, val+oftype(val, 0.5))


rawdata{T<:FixedPoint}(img::Array{Gray{T}, 2}) = convert(Array{Float64, 2}, img)
rawdata(img::Image) = rawdata(data(img)')


# Create dataset consiting of nonzero pixels of face images
# Returns:
#   dat : Matrix(# of nonzeros x # of filenames) - data matrix
#   nonzero_idxs : Vector((i, j)) - nonzero pixel coordinates
function facedata(datadir="../../data/CK/faces_aligned", imsize=IMSIZE)
    filenames = readdir(datadir)
    refmat = rawdata(imread(datadir * "/" * filenames[1]))
    refmat = imresize(refmat, imsize)
    # refmat = refmat ./ 256  -- imresize already does it 
    nonzero_idxs = nonzero_indexes(refmat)
    dat = zeros(length(nonzero_idxs), length(filenames))
    for (i, fname) in enumerate(filenames)
        if i % 500 == 0 println(i, " ", fname) end
        mat = rawdata(imread(datadir * "/" * fname))
        mat = imresize(mat, imsize)
        # mat = mat ./ 256
        dat[:, i] = collect_nonzeros(mat, nonzero_idxs)
    end
    return dat, nonzero_idxs
end


function wplot(W, nzs, padding=10)
    h, w = IMSIZE
    n = size(W, 1)
    rows = int(floor(sqrt(n)))
    cols = int(ceil(n / rows))
    halfpad = div(padding, 2)
    dat = zeros(rows * (h + padding), cols * (w + padding))
    for i=1:n
        wt = W[i, :]
        wt = reshape(wt, length(wt))
        wim = map_nonzeros(IMSIZE, wt, nzs)
        wim = wim ./ (maximum(wim) - minimum(wim))
        r = div(i - 1, cols) + 1
        c = rem(i - 1, cols) + 1
        dat[(r-1)*(h+padding)+halfpad+1 : r*(h+padding)-halfpad,
            (c-1)*(w+padding)+halfpad+1 : c*(w+padding)-halfpad] = wim
    end
    view(dat)
    return dat
end



# REPL





function run(n_hid=1024)
    dat, nzs = facedata()
    n_feat, n_samples = size(dat)
    if isfile("session.jld")
        model, nzs = load("session.jld")
    else
        model = GRBM(n_feat, n_hid, sigma=0.001)
    end
    for i=1:10
        println("meta-iteration #", i)
        fit(model, dat, n_iter=5, n_gibbs=2, lr=0.01)
        save_model(model, nzs, "session.jld")
        println("Sleeping for 20 seconds to cool down")
        sleep(20)
    end
    ## min_images = min(n_hid, 64)
    ## for i=1:min(3, div(n_hid, min_images))
    ##     wplot(model.weights[(i-1)*min_images+1:i*min_images, :], nzs)
    ## end
    return model, nzs
end

# some results
# n_hid=192, n_gibbs=3, n_meta=10 ~ 7 faces of good quality
# n_hid=192, n_gibbs=5, n_meta=10 ~ 4 faces of good quality
# n_hid=256, n_gibbs=1, n_meta=10 ~ no faces at all
# n_hid=256, n_gibbs=3, n_meta=10 ~ 10 faces of almost good quality
# n_hid=256, n_gibbs=5, n_meta=10 ~ 11 faces of almost good quality
# n_hid=256, n_gibbs=10, n_meta=10 ~ 11 faces of almost good quality
# n_hid=256, n_gibbs=5, n_meta=20 ~ 12 faces of almost good quality
# n_hid=128, n_gibbs=3, n_meta=10 ~ all faces of bad quality
# n_hid=192, n_gibbs=3, n_meta=10 ~ 3 faces of a modate quality
# n_hid=256, n_gibbs=3, n_meta=10 ~ 10 faces of moderate quality
# n_hid=320, n_gibbs=3, n_meta=10 ~ 16 faces of almost good qualiy
# n_hid=384, n_gibbs=3, n_meta=10 ~ 17 faces of almost good qualiy
# n_hid=512, n_gibbs=3, n_meta=10 ~ 22 faces of different quality
# n_hid=768, n_gibbs=3, n_meta=10 ~ 25 faces of different quality
# n_hid=1024, n_gibbs=3, n_meta=10 ~ 21 face of different quality
# n_hid=768, n_gibbs=3, n_meta=10, sigma=0.001 ~ almost all faces!
# n_hid=512, n_gibbs=3, n_meta=10, sigma=0.001 ~ all good, but many similar
# n_hid=1024, n_gibbs=3, n_meta=10, sigma=0.001 ~ all good
# n_hid=128, n_gibbs=3, n_meta=10, lr=0.01 ~ 1 good, others similar and bad
# n_hid=128, n_gibbs=10, n_meta=10, lr=0.01 ~ 1 good, others similar and bad
# brbm: n_hid=768, n_gibbs=3, n_meta=10, lr=0.01 ~ 20 good
# grbm: n_hid=1024, n_gibbs=3, n_meta=10 ~ half almost good
# grbm: n_hid=768, n_gibbs=3, n_meta=10 ~ half almost good
# grbm: n_hid=1024, n_gibbs=10, n_meta=10 ~ half almost good
# grbm: n_hid=1024, n_gibbs=10, n_meta=3, lr=0.01 ~ half almost good
#! grbm: n_hid=1024, n_gibbs=3, n_meta=10, sigma=0.001, lr=0.01 ~ all really good!
# brbm: n_hid=1024, n_gibbs=3, n_meta=10, sigma=0.001, lr=0.01 ~ almost the same

# grbm (w/ momentum): n_hid=1024, n_gibbs=2, n_meta=5, sigma=0.001, lr=0.01 ~ 2243



function load_model(filename="session.jld")
    d = load(filename)
    return d["m"], d["nzs"]
end

function save_model(m, nzs, filename="session.jld")
    save(filename, "m", m, "nzs", nzs)
end

## if isinteractive()
##     im = imread("../../data/CK/faces_aligned/S005_001_00000010.png")
##     mat = convert(Array, im)
## end
