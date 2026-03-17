#!/bin/bash
# CSV의 FileName 컬럼에서 파일 경로를 읽어 scp로 다운로드
# 3000개짜리 대형 CSV 제외, 작은 셋 + worst30만 다운로드
# 사용법: bash download_all.sh

SERVER="nsbb@192.168.110.108"
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# 3000개짜리 제외 목록
SKIP_LIST="007.저음질_eval_p 009.한국어_강의_eval_p 010.회의음성_eval_p 012.상담음성_eval_p eval_clean_p eval_other_p"

is_skipped() {
    local name="$1"
    for skip in $SKIP_LIST; do
        [ "$name" = "$skip" ] && return 0
    done
    return 1
}

download_from_csv() {
    local csv_file="$1"
    local out_dir="$2"

    mkdir -p "$out_dir"

    local tmpfile=$(mktemp)
    # BOM 제거 + 헤더 스킵, 첫 번째 컬럼(FileName) 추출
    tail -n +2 "$csv_file" | sed 's/^\xef\xbb\xbf//' | cut -d',' -f1 | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' | grep -v '^$' > "$tmpfile"

    local total=$(wc -l < "$tmpfile")
    echo "  [$total files] → $out_dir"

    if [ "$total" -eq 0 ]; then
        echo "  (no files, skipping)"
        rm -f "$tmpfile"
        return
    fi

    rsync -avz --progress -e "ssh -S /tmp/ssh-ctrl" --files-from="$tmpfile" "$SERVER:/" "$out_dir/"
    local ret=$?

    rm -f "$tmpfile"

    if [ $ret -eq 0 ]; then
        local count=$(find "$out_dir" -type f \( -name '*.wav' -o -name '*.mp3' \) | wc -l)
        echo "  Done: $count files downloaded"
    else
        echo "  ERROR: rsync failed (exit $ret)"
    fi
    echo ""
}

echo "=== Main CSV (small sets only) ==="
for csv in "$BASE_DIR"/*.csv; do
    [ -f "$csv" ] || continue
    name=$(basename "$csv" .csv)
    if is_skipped "$name"; then
        echo "SKIP: $name (3000 files)"
        continue
    fi
    echo "Processing: $name"
    download_from_csv "$csv" "$BASE_DIR/$name"
done

echo ""
echo "=== worst30 ==="
for csv in "$BASE_DIR/worst30"/*.csv; do
    [ -f "$csv" ] || continue
    name=$(basename "$csv" .csv)
    echo "Processing: worst30/$name"
    download_from_csv "$csv" "$BASE_DIR/worst30/$name"
done

echo ""
echo "All done!"
