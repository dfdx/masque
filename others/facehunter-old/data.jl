
using Color
using FixedPointNumbers
using HDF5, JLD

require("interp.jl")


rawdata{T<:FixedPoint}(img::Array{Gray{T}, 2}) = convert(Array{Float64, 2}, img)
rawdata(img::Image) = rawdata(data(img)')


# Create dataset consiting of nonzero pixels of face images
# Returns:
#   dat : Matrix(# of nonzeros x # of filenames) - data matrix
#   nonzero_idxs : Vector((i, j)) - nonzero pixel coordinates
function facedata(datadir=CK_DATA_DIR_FACE_ALIGNED, imsize=IMSIZE)
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


namebase(filename) = join(split(filename, ['_', '.'])[1:3], "_")
labname(nbase) = nbase * "_emotion.txt"
facename(nbase) = nbase * ".png"


function read_resize_image(filename, imsize)
    mat = rawdata(imread(filename))
    return imresize(mat, imsize)
end

function read_aligned(datadir=CK_DATA_DIR, imsize=IMSIZE; labeled_only=false)
    facedir = datadir * FACES_ALIGNED_SUBDIR
    labdir = datadir * LABELES_SUBDIR    
    namebases = (labeled_only ?
                 map(namebase, readdir(labdir)) :
                 map(namebase, readdir(facedir)))
    refimgpath = facedir * "/" * readdir(facedir)[1]
    refmat = read_resize_image(refimgpath, imsize)
    nzs = nonzero_indexes(refmat)
    img_dat = zeros(length(nzs), length(namebases))
    lab_dat = zeros(Int8, length(namebases))
    for (i, nbase) in enumerate(namebases)
        if i % 500 == 0 println("$(i) images processed") end
        mat = read_resize_image(facedir * "/" * facename(nbase), imsize)
        img_dat[:, i] = collect_nonzeros(mat, nzs)
        if isfile(labdir * "/" * labname(nbase))
            lab_dat[i] = int(float(open(readall, labdir * "/" * labname(nbase))))
        else
            lab_dat[i] = -1
        end
    end
    return img_dat, lab_dat, nzs
end
