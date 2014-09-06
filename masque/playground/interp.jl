

# interpolate point at (x, y) from 4 nearby pixel values
function interp_bilinear{T <: Union(Real, Integer)}(dat::Array{T, 2},
                                                    x::Float64, y::Float64,
                                                    x1, x2, y1, y2)
    q11 = dat[y1, x1]         
    q12 = dat[y2, x1]
    q21 = dat[y1, x2]
    q22 = dat[y2, x2]
    if x1 != x2
        r1 = (x2 - x) / (x2 - x1) * q11 + (x - x1) / (x2 - x1) * q21
        r2 = (x2 - x) / (x2 - x1) * q12 + (x - x1) / (x2 - x1) * q22
    else
        # special case of x1 == x2, no interpolation needed
        r1 = q11
        r2 = q12
    end
    if y1 != y2
        r = (y - y1) / (y2 - y1) * r1 + (y2 - y) / (y2 - y1) * r2
    else
        # special case of y1 == y2, no interpolation needed
        r = r1
    end
    if typeof(r) <: Real
        r = round(r)
    end
    r = convert(T, r)
    return r
end


function imresize{T <: Union(Real, Integer)}(dat::Array{T, 2},
                                             new_size::(Int, Int))
    new_dat = similar(dat, new_size)
    h, w = size(dat)
    new_h, new_w = new_size
    for new_j=1:new_w, new_i=1:new_h
        # coordinates in original image
        x = new_j * w / new_w
        y = new_i * h / new_h
        # coordinates of 4 points to interpolate from
        x1, x2 = max(1, floor(x)), min(w, ceil(x))
        y1, y2 = max(1, floor(y)), min(h, ceil(y))
        new_dat[new_i, new_j] = interp_bilinear(dat, x, y, x1, x2, y1, y2)
    end
    return new_dat
end


function imresize{T <: Union(Real, Integer)}(dat::Array{T, 2},
                                             new_size...)
    return imresize(dat, new_size)
end


## if isinteractive()
##     using Images
##     using ImageView
##     im = imread("../../data/CK/faces_aligned/S999_003_00000054.png")
##     dat = convert(Array, im)
## end



using Images, Cairo

function imresize_julia!(resized, original)
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
                           fjo*original[iio+1,ijo+2])
            + fio*((1-fjo)*original[iio+2,ijo+1] +
                   fjo*original[iio+2,ijo+2])
            resized[ir+1,jr+1] = convertsafely(eltype(resized), tmp)
        end
    end
    resized
end
imresize_julia(original, new_size) =
    imresize_julia!(similar(original,
                            new_size), original)
                            
convertsafely{T<:FloatingPoint}(::Type{T}, val) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::Integer) = convert(T, val)
convertsafely{T<:Integer}(::Type{T}, val::FloatingPoint) =
    itrunc(T,
           val+oftype(val, 0.5))
           
           
function imresize_cairo(dat::Array{Uint32, 2}, new_size::(Int, Int))
    cs = CairoImageSurface(dat, 0)
    new_dat = zeros(Uint32, new_size)
    new_cs = CairoImageSurface(new_dat, 0)
    pat = CairoPattern(cs)
    pattern_set_filter(pat, Cairo.FILTER_BILINEAR)
    c = CairoContext(new_cs)
    h, w = size(dat)
    new_h, new_w = new_size
    scale(c, new_h / h, new_w / w)
    set_source(c, pat)
    paint(c)
    return new_cs.data
end

## img = rand(0x00:0xff, 774, 512)
## new_size = (3096, 2048)
## imresize_cairo(convert(Array{Uint32}, img), new_size)
## println("Cairo:")
## @time imresize_cairo(convert(Array{Uint32}, img), new_size)
## imresize_julia(img, new_size)
## println("Julia:")
## @time imresize_julia(img, new_size)
