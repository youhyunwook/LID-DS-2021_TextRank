import os
import sys
import csv
import networkx as nx
from pathlib import Path
from tqdm import tqdm
from collections import Counter

# =========================
# 0. 경로 설정 
# =========================
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE
REPO_ROOT = None

# LID-DS 레포지토리 루트 찾기
for _ in range(6):
    if (ROOT / "dataloader").is_dir() and (ROOT / "algorithms").is_dir():
        REPO_ROOT = ROOT
        break
    ROOT = ROOT.parent

if REPO_ROOT is None:
    print("[ERROR] LID-DS 레포 루트를 찾지 못했습니다.")
    sys.exit(1)

sys.path.insert(0, str(REPO_ROOT))

# LID-DS 모듈
from dataloader.data_loader_2021 import DataLoader2021

# =========================
# 1. 설정값
# =========================

LID_DS_2021_ROOT = r"D:\LID-DS-2021"
OUTPUT_CSV = "LID-DS-2021_Seq2Seq_Dataset.csv"

SEQUENCE_CHUNK_SIZE = 50 
TEXTRANK_TOP_K = 5

SCENARIOS = [
    "Bruteforce_CWE-307",
    "CWE-89-SQL-injection",
    "CVE-2012-2122",
    "CVE-2017-7529",
]

# =========================
# 2. TextRank 알고리즘
# =========================
def get_textrank_keywords(syscall_list, window_size=3, top_k=5):
    """Co-occurrence 기반 TextRank로 키워드 추출"""
    unique_words = list(set(syscall_list))
    if len(unique_words) < top_k:
        return " ".join(unique_words)
    
    graph = nx.Graph()
    graph.add_nodes_from(unique_words)
    
    for i in range(len(syscall_list)):
        for j in range(i + 1, min(i + window_size, len(syscall_list))):
            if syscall_list[i] != syscall_list[j]:
                graph.add_edge(syscall_list[i], syscall_list[j])
    
    try:
        scores = nx.pagerank(graph)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_keywords = [w[0] for w in ranked[:top_k]]
        return " ".join(top_keywords)
    except:
        c = Counter(syscall_list)
        return " ".join([w for w, _ in c.most_common(top_k)])

# =========================
# 3. 데이터 처리 로직
# =========================
def process_scenario(scenario_name, root_path, writer):
    scenario_path = os.path.join(root_path, scenario_name)
    try:
        loader = DataLoader2021(scenario_path)
        recordings = list(loader.test_data())
    except Exception as e:
        print(f"[Skip] {scenario_name} 로드 실패: {e}")
        return 0

    count = 0
    
    for recording in tqdm(recordings, desc=f"{scenario_name}", leave=False):
        try:
            full_syscalls = [sc.name() for sc in recording.syscalls()]
        except:
            continue
            
        if len(full_syscalls) < SEQUENCE_CHUNK_SIZE:
            continue
            
        label = 0
        try:
            meta = recording.metadata()
            if meta.get("exploit", False):
                label = 1
        except:
            pass

        num_chunks = len(full_syscalls) // SEQUENCE_CHUNK_SIZE
        
        for i in range(num_chunks):
            start_idx = i * SEQUENCE_CHUNK_SIZE
            end_idx = start_idx + SEQUENCE_CHUNK_SIZE
            
            chunk = full_syscalls[start_idx:end_idx]
            input_seq_str = " ".join(chunk)
            target_keywords_str = get_textrank_keywords(chunk, top_k=TEXTRANK_TOP_K)
            
            file_name = getattr(recording, "name", "unknown")
            writer.writerow([
                scenario_name, 
                file_name, 
                i, 
                label, 
                input_seq_str, 
                target_keywords_str
            ])
            count += 1
            
    return count

# =========================
# 4. 메인 실행
# =========================
def main():
    print(f"[INFO] LID-DS Root: {LID_DS_2021_ROOT}")
    print(f"[INFO] Output CSV: {OUTPUT_CSV}")
    print(f"[INFO] Seq Chunk Size: {SEQUENCE_CHUNK_SIZE}")
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "file_name", "chunk_idx", "label", "input_sequence", "target_keywords"])
        
        total_rows = 0
        for scenario in SCENARIOS:
            print(f"\nProcessing {scenario}...")
            cnt = process_scenario(scenario, LID_DS_2021_ROOT, writer)
            print(f" -> {cnt} chunks saved.")
            total_rows += cnt
            
    print(f"파일 저장 위치: {os.path.abspath(OUTPUT_CSV)}")

if __name__ == "__main__":

    main()
