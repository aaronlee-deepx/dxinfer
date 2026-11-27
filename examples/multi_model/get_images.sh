#!/usr/bin/env bash
# Download test images for multi_model_demo into this folder:
#   images/image_object_detection/  N photos from COCO val2017      (cocodataset.org, CC BY 4.0 / Flickr terms)
#   images/image_classification/    N ImageNet samples, one per class
#                                    (github.com/EliSchwartz/imagenet-sample-images)
# N is the same for both folders (default 25, 1..50). Files already there are kept.
#
# Usage: ./get_images.sh [-n N] [--list] [target_dir]
#   -n, --count N   images per folder (default 25, max 50)
#   --list          print the URLs that would be downloaded, download nothing
#   target_dir      default: images/ next to this script
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
COUNT=25
LIST_ONLY=0
ROOT=""

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        -n|--count)
            [ $# -ge 2 ] || { echo "[error] $1 needs a value" >&2; exit 1; }
            COUNT="$2"; shift 2 ;;
        --list) LIST_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "[error] unknown option '$1' (see --help)" >&2; exit 1 ;;
        *) ROOT="$1"; shift ;;
    esac
done
ROOT="${ROOT:-$SCRIPT_DIR/images}"

# 50 COCO val2017 image ids and 50 ImageNet samples spread across the 1000 classes (all checked reachable).
COCO_IDS=(000000000139 000000000285 000000000632 000000000724 000000000776 000000000785 000000000802
          000000000872 000000000885 000000001000 000000001268 000000001296 000000001353 000000001425
          000000001490 000000001503 000000001532 000000001584 000000001675 000000001761 000000001818
          000000001993 000000002006 000000002149 000000002153 000000002157 000000002261 000000002299
          000000002431 000000002473 000000002532 000000002587 000000002592 000000002685 000000002923
          000000003156 000000003255 000000003501 000000003553 000000003661 000000003845 000000003934
          000000004134 000000004395 000000004495 000000004765 000000004795 000000005001 000000005037
          000000005060)
IMAGENET=(n01440764_tench n01601694_water_ouzel n01682714_American_chameleon n01742172_boa_constrictor
          n01796340_ptarmigan n01871265_tusker n01983481_American_lobster n02033041_dowitcher n02088364_beagle
          n02093859_Kerry_blue_terrier n02098286_West_Highland_white_terrier n02104365_schipperke
          n02108551_Tibetan_mastiff n02113186_Cardigan n02123597_Siamese_cat n02172182_dung_beetle
          n02281406_sulphur_butterfly n02403003_ox n02480855_gorilla n02504458_African_elephant n02699494_altar
          n02795169_barrel n02841315_binoculars n02927161_butcher_shop n02999410_chain
          n03085013_computer_keyboard n03188531_diaper n03291819_envelope n03417042_garbage_truck
          n03485407_hand-held_computer n03595614_jersey n03680355_Loafer n03761084_microwave
          n03792782_mountain_bike n03868863_oxygen_mask n03920288_Petri_dish n03976467_Polaroid_camera
          n04039381_racket n04131690_saltshaker n04209133_shower_cap n04270147_spatula n04350905_suit
          n04418357_theater_curtain n04487081_trolleybus n04552348_warplane n04613696_yurt
          n07711569_mashed_potato n07760859_custard_apple n09399592_promontory n13054560_bolete)
MAX=${#COCO_IDS[@]}

case "$COUNT" in ''|*[!0-9]*) echo "[error] --count must be a whole number, got '$COUNT'" >&2; exit 1 ;; esac
if [ "$COUNT" -lt 1 ] || [ "$COUNT" -gt "$MAX" ]; then
    echo "[error] --count must be 1..$MAX, got $COUNT" >&2; exit 1
fi

# N indices spread evenly over 0..MAX-1, so a small N still covers varied classes.
pick() {  # pick N -> prints indices
    local n=$1 i
    if [ "$n" -eq 1 ]; then echo 0; return; fi
    for ((i = 0; i < n; i++)); do echo $(( i * (MAX - 1) / (n - 1) )); done
}
mapfile -t IDX < <(pick "$COUNT")

DET_URLS=(); CLS_URLS=()
for i in "${IDX[@]}"; do
    DET_URLS+=("http://images.cocodataset.org/val2017/${COCO_IDS[$i]}.jpg")
    CLS_URLS+=("https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master/${IMAGENET[$i]}.JPEG")
done

if [ "$LIST_ONLY" -eq 1 ]; then
    printf '%s\n' "${DET_URLS[@]}" "${CLS_URLS[@]}"
    exit 0
fi

DET_DIR="$ROOT/image_object_detection"
CLS_DIR="$ROOT/image_classification"
mkdir -p "$DET_DIR" "$CLS_DIR"

if command -v curl >/dev/null 2>&1; then
    fetch() { curl -fsL --retry 3 --connect-timeout 15 -o "$2" "$1" 2>/dev/null; }
    TLS_FAIL_RC=60      # curl: peer certificate cannot be verified
elif command -v wget >/dev/null 2>&1; then
    fetch() { wget -q --tries=3 --timeout=15 -O "$2" "$1"; }
    TLS_FAIL_RC=5       # wget: SSL verification failure
else
    echo "[error] need curl or wget" >&2; exit 1
fi

is_jpeg() { [ -s "$1" ] && [ "$(head -c 3 "$1" | od -An -tx1 | tr -d ' \n')" = "ffd8ff" ]; }

failed=0
tls_failed=0
got_det=0
got_cls=0
get() {  # url dest -> returns 0 when the file is there afterwards
    if is_jpeg "$2"; then echo "  have  $(basename "$2")"; return 0; fi
    local rc=0
    fetch "$1" "$2.part" || rc=$?
    if [ "$rc" -eq 0 ] && is_jpeg "$2.part"; then
        mv "$2.part" "$2"; echo "  got   $(basename "$2")"; return 0
    fi
    rm -f "$2.part"; echo "  FAIL  $1" >&2; failed=$((failed + 1))
    [ "$rc" -eq "$TLS_FAIL_RC" ] && tls_failed=$((tls_failed + 1))
    return 1
}

echo "object detection images ($COUNT) -> $DET_DIR"
for u in "${DET_URLS[@]}"; do get "$u" "$DET_DIR/$(basename "$u")" && got_det=$((got_det + 1)) || true; done
echo "classification images ($COUNT)   -> $CLS_DIR"
for u in "${CLS_URLS[@]}"; do get "$u" "$CLS_DIR/$(basename "$u")" && got_cls=$((got_cls + 1)) || true; done

echo "done: $got_det/$COUNT detection images, $got_cls/$COUNT classification images ($failed failed)"
if [ "$tls_failed" -gt 0 ]; then
    cat >&2 <<'MSG'
[hint] HTTPS certificate verification failed. This network probably inspects TLS traffic with its
       own root certificate, which this Linux system does not trust yet. Install that root certificate:
         sudo cp <your-company-root-ca>.crt /usr/local/share/ca-certificates/ && sudo update-ca-certificates
       or download the images on another machine and copy them into the folders above.
MSG
fi
[ "$failed" -eq 0 ]