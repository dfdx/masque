
require("shape.jl")

function plot_weights(W, nzs, padding=10)
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


flatten(a) = reshape(a, prod(size(a)))
ihist(a) = plt.hist(flatten(a), bins=100)


