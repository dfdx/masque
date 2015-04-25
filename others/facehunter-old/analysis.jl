

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



function svm_accuracy(X, y, k=10)
    svmest(idxs) = svmtrain(y[[idxs]], X[:, idxs])
    svmeval(model, idxs) = svmpredict(model, X[:, idxs])
    return cross_validate(svmest, svmeval, 10, Kfold(length(y), 10))
    
    ## fold_list = collect(Kfold(length(y), k))
    ## for fold_idx in fold_list
        
    ## end
    
end
