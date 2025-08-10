import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

def is_elf_file(filepath: str) -> bool:
    """Проверяет, является ли файл ELF-файлом по магическому числу"""
    try:
        with open(filepath, 'rb') as f:
            return f.read(4) == b'\x7fELF'
    except:
        return False

def process_elf_file(filepath: str, ngram_counts: Dict[bytes, int], max_n: int = 16) -> None:
    """Обрабатывает ELF-файл и обновляет статистику N-грамм"""
    with open(filepath, 'rb') as f:
        data = f.read()
    
    for n in range(1, max_n + 1):
        for i in range(len(data) - n + 1):
            ngram = data[i:i+n]
            ngram_counts[ngram] += 1

def calculate_probabilities(ngram_counts: Dict[bytes, int]) -> List[Tuple[bytes, float]]:
    """Вычисляет вероятности для N-грамм"""
    total = sum(ngram_counts.values())
    return [(ngram, count/total) for ngram, count in ngram_counts.items()]

def save_ngram_stats(output_file: str, ngram_probs: List[Tuple[bytes, float]]) -> None:
    """Сохраняет статистику N-грамм в файл"""
    with open(output_file, 'w') as f:
        for ngram, prob in sorted(ngram_probs, key=lambda x: x[1], reverse=True):
            # Преобразуем байты в читаемый формат (hex)
            hex_str = ' '.join(f'{b:02x}' for b in ngram)
            f.write(f"{hex_str}\t{prob:.10f}\n")

def process_directory(directory: str, output_file: str, max_n: int = 16) -> None:
    """Рекурсивно обрабатывает директорию с ELF-файлами"""
    ngram_counts = defaultdict(int)
    elf_files = 0
    
    for root, _, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if is_elf_file(filepath):
                process_elf_file(filepath, ngram_counts, max_n)
                elf_files += 1
    
    if elf_files == 0:
        print("Не найдено ELF-файлов в указанной директории.")
        return
    
    ngram_probs = calculate_probabilities(ngram_counts)
    save_ngram_stats(output_file, ngram_probs)
    print(f"Обработано {elf_files} ELF-файлов. Результаты сохранены в {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Анализатор N-грамм в ELF-файлах')
    parser.add_argument('directory', help='Директория для рекурсивного поиска ELF-файлов')
    parser.add_argument('output', help='Файл для сохранения статистики N-грамм')
    parser.add_argument('--max_n', type=int, default=16, help='Максимальная длина N-граммы (по умолчанию 16)')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Ошибка: {args.directory} не является директорией.")
        return
    
    process_directory(args.directory, args.output, args.max_n)

if __name__ == "__main__":
    main()
