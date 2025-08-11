import os
import argparse
import mmap
import tempfile
import shutil
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from collections import defaultdict
import xxhash
from datetime import datetime

class ArchiveManager:
    """Класс для управления архивом с несколькими файлами"""
    
    ARCHIVE_HEADER = b'MULTIARC'
    VERSION = 1
    
    def __init__(self, ngram_probs: List[Tuple[bytes, float]]):
        self.compressor = RobustCompressor(ngram_probs)
        
    def create_archive(self, file_paths: List[str], output_path: str):
        """Создание архива из нескольких файлов"""
        # Проверка входных файлов
        valid_files = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"Предупреждение: файл {file_path} не существует, пропускаем")
                continue
            valid_files.append(file_path)
        
        if not valid_files:
            raise ValueError("Нет валидных файлов для архивации")
        
        file_entries = []
        
        with open(output_path, 'wb') as f_out:
            # Записываем заголовок архива (8 байт)
            f_out.write(self.ARCHIVE_HEADER)
            # Записываем версию (2 байта)
            f_out.write(self.VERSION.to_bytes(2, 'big'))
            
            # Зарезервируем место для таблицы файлов (8 байт)
            table_position = f_out.tell()
            f_out.write(b'\x00\x00\x00\x00')  # Позиция таблицы
            f_out.write(b'\x00\x00\x00\x00')  # Размер таблицы
            
            # Сжимаем каждый файл и добавляем в архив
            for file_path in valid_files:
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)
                mod_time = os.path.getmtime(file_path)
                mod_time_str = datetime.fromtimestamp(mod_time).isoformat()
                
                # Временный файл для сжатых данных
                temp_path = output_path + '.tmp'
                
                try:
                    # Сжимаем файл
                    self.compressor.compress(file_path, temp_path)
                    comp_size = os.path.getsize(temp_path)
                    
                    # Читаем сжатые данные
                    with open(temp_path, 'rb') as f_temp:
                        compressed = f_temp.read()
                    
                    # Записываем сжатые данные
                    data_start = f_out.tell()
                    f_out.write(compressed)
                    data_end = f_out.tell()
                    
                    # Добавляем информацию о файле
                    file_entries.append({
                        'name': file_name,
                        'original_size': file_size,
                        'compressed_size': comp_size,
                        'start': data_start,
                        'end': data_end,
                        'modified': mod_time_str,
                        'compression_ratio': (1 - comp_size/file_size) * 100
                    })
                finally:
                    # Удаляем временный файл, если он существует
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            
            # Записываем таблицу файлов
            table_start = f_out.tell()
            table_data = json.dumps(file_entries, ensure_ascii=False).encode('utf-8')
            f_out.write(table_data)
            table_end = f_out.tell()
            
            # Возвращаемся и обновляем указатель на таблицу
            f_out.seek(table_position)
            f_out.write(table_start.to_bytes(4, 'big', signed=False))
            f_out.write((table_end - table_start).to_bytes(4, 'big', signed=False))
        
        print(f"Архив {output_path} успешно создан, сжато {len(file_entries)} файлов")
    def add_files(self, archive_path: str, new_files: List[str]):
        """Добавление файлов в существующий архив"""
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Архив не найден: {archive_path}")

        # Проверяем новые файлы
        valid_new_files = []
        for file_path in new_files:
            if not os.path.exists(file_path):
                print(f"Файл не найден: {file_path}, пропускаем")
                continue
            valid_new_files.append(file_path)
        
        if not valid_new_files:
            print("Нет валидных файлов для добавления")
            return
        
        # Создаем временную директорию
        temp_dir = tempfile.mkdtemp()
        try:
            # Сначала читаем всю информацию о существующем архиве
            with open(archive_path, 'rb') as f:
                header = f.read(8)
                if header != self.ARCHIVE_HEADER:
                    raise ValueError("Неверный формат архива")
                
                version = int.from_bytes(f.read(2), 'big')
                if version != self.VERSION:
                    raise ValueError(f"Неверная версия архива: {version}")
                
                table_pos = int.from_bytes(f.read(4), 'big')
                table_size = int.from_bytes(f.read(4), 'big')
                
                f.seek(table_pos)
                table_data = f.read(table_size)
                existing_entries = json.loads(table_data.decode('utf-8'))
            
            # Проверяем дубликаты
            existing_names = {e['name'] for e in existing_entries}
            files_to_add = []
            
            for file_path in valid_new_files:
                file_name = os.path.basename(file_path)
                if file_name in existing_names:
                    print(f"Файл {file_name} уже существует в архиве, пропускаем")
                    continue
                files_to_add.append(file_path)
            
            if not files_to_add:
                print("Нет новых файлов для добавления (все уже существуют в архиве)")
                return
            
            # Собираем все файлы для нового архива
            all_files_paths = []
            
            # 1. Сначала добавляем новые файлы
            all_files_paths.extend(files_to_add)
            
            # 2. Затем добавляем существующие файлы (извлекаем их во временную директорию)
            with open(archive_path, 'rb') as f:
                for entry in existing_entries:
                    temp_file_path = os.path.join(temp_dir, entry['name'])
                    with open(temp_file_path, 'wb') as f_out:
                        f.seek(entry['start'])
                        compressed_data = f.read(entry['end'] - entry['start'])
                        f_out.write(compressed_data)
                    all_files_paths.append(temp_file_path)
            
            # Создаем новый архив
            temp_archive_path = os.path.join(temp_dir, 'temp.arc')
            self.create_archive(all_files_paths, temp_archive_path)
            
            # Заменяем старый архив новым
            shutil.move(temp_archive_path, archive_path)
            print(f"Успешно добавлено {len(files_to_add)} файлов в архив")
        
        except Exception as e:
            raise ValueError(f"Ошибка при добавлении файлов: {str(e)}")
        finally:
            # Удаляем временную директорию
            shutil.rmtree(temp_dir, ignore_errors=True)

    def list_files(self, archive_path: str):
        """Просмотр содержимого архива"""
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Архив не найден: {archive_path}")
        
        try:
            with open(archive_path, 'rb') as f:
                # Читаем заголовок (8 байт)
                header = f.read(8)
                if header != self.ARCHIVE_HEADER:
                    raise ValueError("Неверный формат архива (некорректный заголовок)")
                
                # Читаем версию (2 байта)
                version_bytes = f.read(2)
                if len(version_bytes) != 2:
                    raise ValueError("Не удалось прочитать версию архива")
                    
                version = int.from_bytes(version_bytes, 'big', signed=False)
                if version != self.VERSION:
                    raise ValueError(f"Неверная версия архива: {version} (ожидалось {self.VERSION})")
                
                # Читаем позицию и размер таблицы (по 4 байта)
                table_pos_bytes = f.read(4)
                table_size_bytes = f.read(4)
                
                if len(table_pos_bytes) != 4 or len(table_size_bytes) != 4:
                    raise ValueError("Не удалось прочитать метаданные таблицы файлов")
                    
                table_pos = int.from_bytes(table_pos_bytes, 'big', signed=False)
                table_size = int.from_bytes(table_size_bytes, 'big', signed=False)
                
                # Переходим к таблице файлов
                f.seek(table_pos)
                table_data = f.read(table_size)
                
                try:
                    file_entries = json.loads(table_data.decode('utf-8'))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Ошибка чтения таблицы файлов: {str(e)}")
                
                # Выводим информацию о файлах
                print("\nФайлы в архиве:")
                print("-" * 80)
                print(f"{'Имя файла':<30} | {'Размер':>10} | {'Сжатый':>10} | {'Коэфф.':>7}% | Дата изменения")
                print("-" * 80)
                
                for entry in file_entries:
                    print(f"{entry['name'][:30]:<30} | "
                        f"{entry['original_size']:>10} | "
                        f"{entry['compressed_size']:>10} | "
                        f"{entry['compression_ratio']:>6.2f}% | "
                        f"{entry['modified']}")
        
        except Exception as e:
            raise ValueError(f"Ошибка чтения архива: {str(e)}")

    def extract_file(self, archive_path: str, file_name: str, output_dir: str = '.'):
        """Извлечение одного файла из архива"""
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Архив не найден: {archive_path}")
        
        with open(archive_path, 'rb') as f:
            header = f.read(8)
            if header[:8] != self.ARCHIVE_HEADER:
                raise ValueError("Неверный формат архива")
            
            f.seek(10)  # Пропускаем версию
            table_pos = int.from_bytes(f.read(4), 'big')
            table_size = int.from_bytes(f.read(4), 'big')
            
            f.seek(table_pos)
            table_data = f.read(table_size)
            file_entries = json.loads(table_data.decode('utf-8'))
            
            found = False
            for entry in file_entries:
                if entry['name'] == file_name:
                    found = True
                    output_path = os.path.join(output_dir, file_name)
                    
                    # Читаем сжатые данные
                    f.seek(entry['start'])
                    compressed_data = f.read(entry['end'] - entry['start'])
                    
                    # Записываем во временный файл
                    temp_path = archive_path + '.tmp'
                    with open(temp_path, 'wb') as f_temp:
                        f_temp.write(compressed_data)
                    
                    # Распаковываем
                    self.compressor.decompress(temp_path, output_path)
                    os.remove(temp_path)
                    
                    # Восстанавливаем дату модификации
                    mod_time = datetime.fromisoformat(entry['modified']).timestamp()
                    os.utime(output_path, (mod_time, mod_time))
                    
                    print(f"Файл {file_name} успешно извлечен в {output_path}")
                    break
            
            if not found:
                raise ValueError(f"Файл {file_name} не найден в архиве")
    
    def extract_all(self, archive_path: str, output_dir: str = '.'):
        """Извлечение всех файлов из архива"""
        if not os.path.exists(archive_path):
            raise FileNotFoundError(f"Архив не найден: {archive_path}")
        
        with open(archive_path, 'rb') as f:
            header = f.read(8)
            if header[:8] != self.ARCHIVE_HEADER:
                raise ValueError("Неверный формат архива")
            
            f.seek(10)  # Пропускаем версию
            table_pos = int.from_bytes(f.read(4), 'big')
            table_size = int.from_bytes(f.read(4), 'big')
            
            f.seek(table_pos)
            table_data = f.read(table_size)
            file_entries = json.loads(table_data.decode('utf-8'))
            
            os.makedirs(output_dir, exist_ok=True)
            
            for entry in file_entries:
                try:
                    file_name = entry['name']
                    output_path = os.path.join(output_dir, file_name)
                    
                    # Читаем сжатые данные
                    f.seek(entry['start'])
                    compressed_data = f.read(entry['end'] - entry['start'])
                    
                    # Записываем во временный файл
                    temp_path = archive_path + '.tmp'
                    with open(temp_path, 'wb') as f_temp:
                        f_temp.write(compressed_data)
                    
                    # Распаковываем
                    self.compressor.decompress(temp_path, output_path)
                    os.remove(temp_path)
                    
                    # Восстанавливаем дату модификации
                    mod_time = datetime.fromisoformat(entry['modified']).timestamp()
                    os.utime(output_path, (mod_time, mod_time))
                    
                    print(f"Файл {file_name} успешно извлечен")
                except Exception as e:
                    print(f"Ошибка при извлечении {entry['name']}: {str(e)}")
  

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
    parser = argparse.ArgumentParser(description='Многофайловый архиватор с оптимизациями')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Команда создания архива
    create_parser = subparsers.add_parser('create', help='Создать новый архив')
    create_parser.add_argument('archive', help='Имя архива')
    create_parser.add_argument('files', nargs='+', help='Файлы для добавления в архив')
    create_parser.add_argument('ngrams', help='Файл с 8-граммами')
    
    # Команда просмотра содержимого
    list_parser = subparsers.add_parser('list', help='Просмотреть содержимое архива')
    list_parser.add_argument('archive', help='Имя архива')
    list_parser.add_argument('ngrams', help='Файл с 8-граммами')
    
    # Команда извлечения одного файла
    extract_parser = subparsers.add_parser('extract', help='Извлечь файл из архива')
    extract_parser.add_argument('archive', help='Имя архива')
    extract_parser.add_argument('file', help='Имя файла для извлечения')
    extract_parser.add_argument('ngrams', help='Файл с 8-граммами')
    extract_parser.add_argument('--output', '-o', default='.', help='Целевая директория')
    
    # Команда извлечения всех файлов
    extract_all_parser = subparsers.add_parser('extract-all', help='Извлечь все файлы из архива')
    extract_all_parser.add_argument('archive', help='Имя архива')
    extract_all_parser.add_argument('ngrams', help='Файл с 8-граммами')
    extract_all_parser.add_argument('--output', '-o', default='.', help='Целевая директория')
    
    # Команда добавления файлов
    add_parser = subparsers.add_parser('add', help='Добавить файлы в архив')
    add_parser.add_argument('archive', help='Имя архива')
    add_parser.add_argument('files', nargs='+', help='Файлы для добавления')
    add_parser.add_argument('ngrams', help='Файл с 8-граммами')
    
    args = parser.parse_args()
    
    try:
        ngrams = load_ngrams(args.ngrams)
        manager = ArchiveManager(ngrams)
        
        if args.command == 'create':
            print("Создание архива...")
            manager.create_archive(args.files, args.archive)
            print(f"Архив {args.archive} успешно создан")
            manager.list_files(args.archive)
            
        elif args.command == 'list':
            manager.list_files(args.archive)
            
        elif args.command == 'extract':
            print(f"Извлечение файла {args.file}...")
            manager.extract_file(args.archive, args.file, args.output)
            
        elif args.command == 'extract-all':
            print("Извлечение всех файлов...")
            manager.extract_all(args.archive, args.output)
            
        elif args.command == 'add':
            print("Добавление файлов в архив...")
            manager.add_files(args.archive, args.files)
            manager.list_files(args.archive)
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
