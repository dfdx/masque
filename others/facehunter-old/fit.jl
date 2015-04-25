
const N_HID = 1024
const SIGMA = 0.001


function save_model(m, nzs)
    save(MODEL_FILE, "m", m)
    save(NZS_FILE, "nzs", nzs)
end

function load_model()
   load(MODEL_FILE)["m"], load(NZS_FILE)["nzs"] 
end


function fit_and_save()
    @time dat, _, nzs = read_aligned(labeled_only=true)
    n_feat, n_samples = size(dat)    
    m = GRBM(n_feat, N_HID, sigma=SIGMA)    
    for i=1:8
        println("meta-iteration #", i)
        @time fit(m, dat, n_iter=10, n_gibbs=2, lr=0.01)
    end
    save_model(m, nzs)
end
