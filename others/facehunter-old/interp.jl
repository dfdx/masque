
using Images
using ImageView

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
            resized[ir+1,jr+1] = convertsafely(eltype(resized), tmp)
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
           
