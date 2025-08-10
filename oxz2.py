import os
import argparse
import mmap
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
import xxhash

class RobustCompressor:
    def __init__(self, ngram_probs: List[Tuple[bytes, float]]):
        self.ngram_size = 8
        self.ngram_dict = {ngram: idx for idx, (ngram, _) in enumerate(ngram_probs[:65535])}
        self.inverse_dict = {idx: ngram for idx, (ngram, _) in enumerate(ngram_probs[:65535])}
        
        self.window_size = 16384
        self.min_match = 4
        self.max_match = 258
        self.min_repeat_length = 16
        self.max_repeat_length = 65535
        self.zero_optimization_threshold = 32
        
        self.MARKER_RAW_BYTE = 0xFFFE
        self.MARKER_LZ77 = 0xFFFF
        self.MARKER_ZEROS = 0xFFFD
        self.MARKER_REPEAT_BYTE = 0xFFFC
        self.MARKER_OPTIMIZED_ZEROS = 0xFFFB
        
        self.hash_table = defaultdict(list)
        self.hash_func = xxhash.xxh32
        
        self.simd_width = 32
        self.chunk_size = 1 << 20  # 1MB chunks
        self.max_workers = os.cpu_count() or 4

    def compress(self, input_path: str, output_path: str):
        temp_path = output_path + ".tmp"
        self._parallel_compress(input_path, temp_path)
        self._optimize_zeros(temp_path, output_path)
        
        try:
            os.remove(temp_path)
        except OSError:
            pass

        orig_size = os.path.getsize(input_path)
        comp_size = os.path.getsize(output_path)
        ratio = (1 - comp_size/orig_size) * 100
        print(f"Сжатие завершено. Коэффициент: {ratio:.2f}%")

    def _parallel_compress(self, input_path: str, output_path: str):
        file_size = os.path.getsize(input_path)
        chunks = [(i, min(i + self.chunk_size, file_size)) 
                 for i in range(0, file_size, self.chunk_size)]
        
        with open(output_path, 'wb') as f_out:
            f_out.write(b'NG')
            f_out.write(len(self.ngram_dict).to_bytes(2, 'big'))
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i, (start, end) in enumerate(chunks):
                futures.append(
                    executor.submit(
                        self._compress_chunk,
                        input_path, output_path, start, end, i
                    )
                )
            
            for future in as_completed(futures):
                future.result()

    def _compress_chunk(self, input_path: str, output_path: str, 
                      start: int, end: int, chunk_id: int):
        with open(input_path, 'rb') as f_in:
            # Используем mmap только для чтения
            with mmap.mmap(f_in.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                chunk_data = mm[start:end]
                if not chunk_data:
                    return
                
                # Копируем данные в bytearray, чтобы избежать проблем с указателями
                buffer = bytearray(chunk_data)
                compressed = bytearray()
                history = bytearray()
                
                self._build_hash_table(buffer)
                
                pos = 0
                while pos < len(buffer):
                    repeat_info = self._find_repeats(buffer, pos)
                    if repeat_info and repeat_info[1] >= self.min_repeat_length:
                        byte, count = repeat_info
                        count = min(int(count), len(buffer) - pos)
                        
                        if byte == 0:
                            compressed.extend(self.MARKER_ZEROS.to_bytes(2, 'big'))
                        else:
                            compressed.extend(self.MARKER_REPEAT_BYTE.to_bytes(2, 'big'))
                            compressed.append(byte)
                        
                        compressed.extend(count.to_bytes(2, 'big'))
                        history.extend(bytes([byte]) * count)
                        pos += count
                        continue
                    
                    if pos + self.ngram_size <= len(buffer):
                        ngram = bytes(buffer[pos:pos+self.ngram_size])
                        if ngram in self.ngram_dict:
                            code = self.ngram_dict[ngram]
                            compressed.extend(code.to_bytes(2, 'big'))
                            history.extend(ngram)
                            pos += self.ngram_size
                            continue
                    
                    match = self._find_best_match(history, buffer, pos)
                    if match and match[1] >= self.min_match:
                        offset, length = match
                        compressed.extend(self.MARKER_LZ77.to_bytes(2, 'big'))
                        compressed.extend(int(offset).to_bytes(2, 'big'))
                        compressed.extend(int(length).to_bytes(1, 'big'))
                        
                        matched_data = buffer[pos:pos+length]
                        history.extend(matched_data)
                        pos += length
                    else:
                        compressed.extend(self.MARKER_RAW_BYTE.to_bytes(2, 'big'))
                        compressed.append(buffer[pos])
                        history.append(buffer[pos])
                        pos += 1
                    
                    if len(history) > self.window_size:
                        del history[:len(history)-self.window_size]
                
                # Записываем сжатые данные в файл
                with open(output_path, 'r+b') as f_out:
                    f_out.seek(0, os.SEEK_END)
                    f_out.write(compressed)

    def _build_hash_table(self, data: bytearray):
        self.hash_table.clear()
        for i in range(len(data) - self.min_match + 1):
            chunk = bytes(data[i:i+self.min_match])
            h = self.hash_func(chunk).intdigest()
            self.hash_table[h].append(i)

    def _find_best_match(self, history: bytearray, buffer: bytearray, pos: int) -> Optional[Tuple[int, int]]:
        if not history or pos + self.min_match > len(buffer):
            return None
            
        chunk = bytes(buffer[pos:pos+self.min_match])
        h = self.hash_func(chunk).intdigest()
        candidates = self.hash_table.get(h, [])
        
        if not candidates:
            return None
            
        best_offset, best_length = 0, self.min_match - 1
        max_possible = min(self.max_match, len(buffer) - pos)
        
        for candidate_pos in candidates[:1000]:
            if candidate_pos >= pos:
                continue
                
            offset = pos - candidate_pos
            if offset > self.window_size:
                continue
                
            match_len = 0
            while (match_len < max_possible and 
                   candidate_pos + match_len < len(history) and 
                   pos + match_len < len(buffer) and 
                   history[candidate_pos + match_len] == buffer[pos + match_len]):
                match_len += 1
            
            if match_len > best_length:
                best_length = match_len
                best_offset = offset
                if best_length >= self.max_match:
                    break
        
        return (int(best_offset), int(best_length)) if best_length >= self.min_match else None

    def _find_repeats(self, data: bytearray, pos: int) -> Optional[Tuple[int, int]]:
        """Поиск повторений без использования SIMD"""
        if pos + self.min_repeat_length > len(data):
            return None
            
        first_byte = data[pos]
        count = 1
        while pos + count < len(data) and data[pos + count] == first_byte:
            count += 1
            if count >= self.max_repeat_length:
                break
        
        return (first_byte, count) if count >= self.min_repeat_length else None

    def _optimize_zeros(self, input_path: str, output_path: str):
        """Оптимизация нулей без использования mmap"""
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            # Читаем файл целиком в память
            buffer = bytearray(f_in.read())
            i = 0
            n = len(buffer)
            
            # Копируем заголовок (первые 4 байта)
            if n >= 4:
                f_out.write(buffer[:4])
                i = 4
            
            while i < n:
                if i + 1 < n:
                    marker = (buffer[i] << 8) | buffer[i+1]
                    
                    if marker == self.MARKER_RAW_BYTE and i + 2 < n and buffer[i+2] == 0:
                        zero_count = 1
                        j = i + 3
                        while j + 2 < n and zero_count < self.max_repeat_length:
                            next_marker = (buffer[j] << 8) | buffer[j+1]
                            if next_marker == self.MARKER_RAW_BYTE and buffer[j+2] == 0:
                                zero_count += 1
                                j += 3
                            else:
                                break
                        
                        if zero_count >= self.zero_optimization_threshold:
                            f_out.write(self.MARKER_OPTIMIZED_ZEROS.to_bytes(2, 'big'))
                            f_out.write(zero_count.to_bytes(2, 'big'))
                            i = j
                            continue
                    
                    elif marker == self.MARKER_ZEROS and i + 4 < n:
                        length = (buffer[i+2] << 8) | buffer[i+3]
                        if length >= self.zero_optimization_threshold * 2:
                            total_length = length
                            j = i + 4
                            while j + 4 < n and total_length < self.max_repeat_length:
                                next_marker = (buffer[j] << 8) | buffer[j+1]
                                if next_marker == self.MARKER_ZEROS:
                                    next_length = (buffer[j+2] << 8) | buffer[j+3]
                                    total_length += next_length
                                    j += 4
                                else:
                                    break
                            
                            if total_length > length:
                                f_out.write(self.MARKER_OPTIMIZED_ZEROS.to_bytes(2, 'big'))
                                f_out.write(total_length.to_bytes(2, 'big'))
                                i = j
                                continue
                
                # Если не нашли оптимизируемую последовательность, копируем как есть
                if i < n:
                    f_out.write(buffer[i:i+1])
                    i += 1

    def decompress(self, input_path: str, output_path: str):
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            header = f_in.read(4)
            if len(header) != 4 or header[:2] != b'NG':
                raise ValueError("Неверный формат файла")
            
            dict_size = int.from_bytes(header[2:4], 'big')
            output_buffer = bytearray()
            
            while True:
                marker = f_in.read(2)
                if not marker:
                    break
                
                marker_int = int.from_bytes(marker, 'big')
                
                if marker_int == self.MARKER_RAW_BYTE:
                    byte = f_in.read(1)
                    if not byte:
                        break
                    output_buffer.append(byte[0])
                    f_out.write(byte)
                
                elif marker_int == self.MARKER_LZ77:
                    offset_bytes = f_in.read(2)
                    length_byte = f_in.read(1)
                    if not offset_bytes or not length_byte:
                        raise ValueError("Неожиданный конец файла")
                    
                    offset = int.from_bytes(offset_bytes, 'big')
                    length = length_byte[0]
                    
                    if offset == 0 or offset > len(output_buffer):
                        output_buffer.append(0)
                        f_out.write(b'\x00')
                        continue
                    
                    pos = len(output_buffer) - offset
                    copy_data = output_buffer[pos:pos+length]
                    if len(copy_data) < length:
                        copy_data += bytes(length - len(copy_data))
                    output_buffer.extend(copy_data)
                    f_out.write(copy_data)
                
                elif marker_int == self.MARKER_ZEROS:
                    length_bytes = f_in.read(2)
                    if not length_bytes:
                        raise ValueError("Неожиданный конец файла")
                    length = int.from_bytes(length_bytes, 'big')
                    zeros = bytes(length)
                    output_buffer.extend(zeros)
                    f_out.write(zeros)
                
                elif marker_int == self.MARKER_REPEAT_BYTE:
                    byte_data = f_in.read(1)
                    length_bytes = f_in.read(2)
                    if not byte_data or not length_bytes:
                        raise ValueError("Неожиданный конец файла")
                    
                    byte = byte_data[0]
                    length = int.from_bytes(length_bytes, 'big')
                    repeated = bytes([byte]) * length
                    output_buffer.extend(repeated)
                    f_out.write(repeated)
                
                elif marker_int == self.MARKER_OPTIMIZED_ZEROS:
                    length_bytes = f_in.read(2)
                    if not length_bytes:
                        raise ValueError("Неожиданный конец файла")
                    length = int.from_bytes(length_bytes, 'big')
                    zeros = bytes(length)
                    output_buffer.extend(zeros)
                    f_out.write(zeros)
                
                else:
                    if marker_int < dict_size and marker_int in self.inverse_dict:
                        ngram = self.inverse_dict[marker_int]
                        output_buffer.extend(ngram)
                        f_out.write(ngram)
                    else:
                        zeros = b'\x00' * self.ngram_size
                        output_buffer.extend(zeros)
                        f_out.write(zeros)

def load_ngrams(file_path: str) -> List[Tuple[bytes, float]]:
    ngrams = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                hex_str, prob = parts
                try:
                    ngram = bytes.fromhex(hex_str)
                    if len(ngram) == 8:
                        ngrams.append((ngram, float(prob)))
                except ValueError:
                    continue
    return sorted(ngrams, key=lambda x: x[1], reverse=True)

def main():
    parser = argparse.ArgumentParser(description='Высокопроизводительный компрессор с оптимизациями')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    compress_parser = subparsers.add_parser('compress', help='Сжать файл')
    compress_parser.add_argument('input', help='Входной файл')
    compress_parser.add_argument('output', help='Сжатый файл')
    compress_parser.add_argument('ngrams', help='Файл с 8-граммами')
    
    decompress_parser = subparsers.add_parser('decompress', help='Распаковать файл')
    decompress_parser.add_argument('input', help='Сжатый файл')
    decompress_parser.add_argument('output', help='Восстановленный файл')
    decompress_parser.add_argument('ngrams', help='Файл с 8-граммами')
    
    args = parser.parse_args()
    
    try:
        ngrams = load_ngrams(args.ngrams)
        compressor = RobustCompressor(ngrams)
        
        if args.command == 'compress':
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Файл не найден: {args.input}")
            
            print("Сжатие начато...")
            compressor.compress(args.input, args.output)
            
        elif args.command == 'decompress':
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Файл не найден: {args.input}")
            
            print("Распаковка начата...")
            compressor.decompress(args.input, args.output)
            print(f"Распаковка завершена. Результат: {args.output}")
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
