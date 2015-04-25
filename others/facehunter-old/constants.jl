
# global constants used by more than one module

const DATA_DIR = expanduser("~/data/")
const CK_DATA_DIR_FACE_ALIGNED = DATA_DIR * "/CK/faces_aligned/"
const CK_DATA_DIR = expanduser("~/data/CK/")
const FACES_ALIGNED_SUBDIR = "faces_aligned"
const LABELES_SUBDIR = "labels"
const IMSIZE = (96, 96)
const MODEL_FILE = DATA_DIR * "facehunter/model.jld"
const NZS_FILE = DATA_DIR * "facehunter/nzs.jld"
