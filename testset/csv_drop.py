import pandas as pd
import os

def cleanup_csv_files(csv_list):
    for csv_file in csv_list:
        if not os.path.exists(csv_file):
            print(f"⚠️ 파일을 찾을 수 없습니다: {csv_file}")
            continue
        
        try:
            # 1. CSV 읽기
            df = pd.read_csv(csv_file)
            
            # 2. 필요한 컬럼만 선택 (FileName, gt)
            # 만약 컬럼명에 공백이 섞여있을 수 있으니 strip() 처리 포함
            df.columns = [c.strip() for c in df.columns]
            
            if 'FileName' in df.columns and 'gt' in df.columns:
                df_cleaned = df[['FileName', 'gt']]
                
                # 3. 동일한 파일명으로 저장 (덮어쓰기)
                df_cleaned.to_csv(csv_file, index=False, encoding='utf-8-sig')
                print(f"✅ 정리 완료: {csv_file}")
            else:
                print(f"❌ 필수 컬럼 누락 (FileName 또는 gt 없음): {csv_file}")
                
        except Exception as e:
            print(f"🔥 에러 발생 ({csv_file}): {e}")

if __name__ == "__main__":
    # 정리하고 싶은 CSV 파일 리스트를 입력하세요.
    target_csvs = ["/nas04/nlp_sk/STT/data/test/modelhouse_2m_noheater.csv", "/nas04/nlp_sk/STT/data/test/modelhouse_2m.csv", "/nas04/nlp_sk/STT/data/test/modelhouse_3m.csv"] 
    cleanup_csv_files(target_csvs)