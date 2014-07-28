 
using Images
using ImageView
using MiLK.NNet.RBM


function nonzero_indexes(mat::Matrix{Uint8})
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


function collect_nonzeros(mat::Matrix{Uint8}, idxs::Array{(Int, Int)})
    arr = zeros(Uint8, length(idxs))
    k = 1
    for (i, j) in idxs
        arr[k] = mat[i, j]
        k += 1
    end
    return arr
end


function map_nonzeros(matsize::(Int, Int), arr::Array{Uint8, 1},
                      idxs::Array{(Int, Int)})
    mat = zeros(Uint8, matsize)
    k = 1
    for (i, j) in idxs
        mat[i, j] = arr[k]
        k += 1
    end
    return mat
end


## function readfaces(datadir="../../data/CK/faces_aligned", imsize=(256, 256))
##     @task begin
##         filenames = readdir(datadir)
##         # mats = zeros(imsize..., length(filenames))
##         for fname in filenames
##             produce(convert(Array, imread(datadir * "/" * fname)))
##         end
##     end    
## end


# Create dataset consiting of nonzero pixels of face images
# Returns:
#   dat : Matrix(# of filenames x # of nonzeros) - data matrix
#   nonzero_idxs : Vector((i, j)) - nonzero pixel coordinates
function facedata(datadir="../../data/CK/faces_aligned", imsize=(256, 256))
    filenames = readdir(datadir)
    dat = zeros(imsize..., length(filenames))
    refmat = convert(Array, imread(datadir * "/" * filenames[1]))
    nonzero_idxs = nonzero_indexes(refmat)
    for (i, fname) in enumerate(filenames)
        println(i, " ", fname)
        mat = convert(Array, imread(datadir * "/" * fname))
        dat[:, :, i] = collect_nonzeros(dat, nonzero_indexes)
    end
    return dat, nonzero_idxs
end


## function faceimdata(datadir="../../data/CK/faces_aligned", imsize=(256, 256))
##     filenames = readdir(datadir)
##     dat = zeros(imsize..., length(filenames))
##     for (i, fname) in enumerate(filenames)
##         dat[:, :, i] = convert(Array, imread(datadir * "/" * fname))
##     end
##     return dat
## end



# REPL

if isinteractive()
    im = imread("../../data/CK/faces_aligned/S005_001_00000010.png")
    mat = convert(Array, im)
end
