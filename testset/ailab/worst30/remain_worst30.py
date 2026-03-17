import pandas as pd
import os

def filter_top_cer_and_save(file_path, top_count=30):
    """
    CSV 파일을 읽어 CER이 높은 순으로 상위 N개를 추출하고, 
    평균 CER을 계산하여 파일명에 포함시켜 저장하는 함수.
    """
    # 1. CSV 파일 읽기
    df = pd.read_csv(file_path)

    # 2. 'cer' 컬럼을 숫자로 변환 (데이터에 문자가 섞여 있을 경우 대비)
    df['cer'] = pd.to_numeric(df['cer'], errors='coerce')
    
    # 3. CER 기준 내림차순 정렬 (높은 값이 위로)
    df_sorted = df.sort_values(by='cer', ascending=False)

    # 4. 상위 30개(top_count)만 남기기
    df_top = df_sorted.head(top_count)

    # 5. 상위 30개 데이터의 평균 CER 계산
    avg_cer = df_top['cer'].mean()

    # 6. 새 파일명 생성 (원본 파일명_평균CER.csv)
    # 파일명과 확장자 분리 (예: 'results.csv' -> 'results')
    file_name_no_ext = os.path.splitext(file_path)[0]
    
    # 소수점 4째자리까지 파일명에 포함 (예: results_top30_avg_0.4567.csv)
    output_file = f"{file_name_no_ext}_top{top_count}_avg_{avg_cer*100:.2f}.csv"

    # 7. 결과 저장 (한글 깨짐 방지를 위해 utf-8-sig 사용)
    df_top.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"처리 완료!")
    print(f" - 추출된 데이터 수: {len(df_top)}개")
    print(f" - 상위 데이터 평균 CER: {avg_cer*100:.2f}")
    print(f" - 저장된 파일명: {output_file}")

    return output_file

# --- 사용 예시 ---

target_file_list = [
    '/nas04/nlp_sk/STT/data/test/sample30/007_results_5.76.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/7F_HJY_results_9.27.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/7F_KSK_results_2.66.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/009_results_9.97.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/010_results_10.76.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/012_results_4.86.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/eval_clean_p_results_10.07.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/eval_other_p_results_9.54.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/modelhouse_2m_noheater_results_3.59.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/modelhouse_2m_results_8.51.csv',
    '/nas04/nlp_sk/STT/data/test/sample30/modelhouse_3m_results_15.72.csv'
]

for target_file in target_file_list:
    saved_path = filter_top_cer_and_save(target_file)