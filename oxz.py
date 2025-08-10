import os
import argparse
from typing import Dict, List, Tuple, Optional, Union

class RobustCompressor:
    def __init__(self, ngram_probs: List[Tuple[bytes, float]]):
        # Словарь 8-грамм
        self.ngram_size = 8
        self.ngram_dict = {ngram: idx for idx, (ngram, _) in enumerate(ngram_probs[:65535])}
        self.inverse_dict = {idx: ngram for idx, (ngram, _) in enumerate(ngram_probs[:65535])}
        
        # Параметры LZ77
        self.window_size = 16280
        self.min_match = 2
        self.max_match = 255
        
        # Параметры для сжатия повторений
        self.min_repeat_length = 16
        self.max_repeat_length = 65535
        self.zero_optimization_threshold = 16
        
        # Специальные маркеры
        self.MARKER_RAW_BYTE = 0xFFFE
        self.MARKER_LZ77 = 0xFFFF
        self.MARKER_ZEROS = 0xFFFD
        self.MARKER_REPEAT_BYTE = 0xFFFC
        self.MARKER_OPTIMIZED_ZEROS = 0xFFFB  # Новый маркер для оптимизированных нулей

    def compress(self, input_path: str, output_path: str):
        """Двухэтапное сжатие с постобработкой"""
        # Этап 1: Базовое сжатие
        temp_path = output_path + ".tmp"
        self._basic_compress(input_path, temp_path)
        
        # Этап 2: Оптимизация нулей
        self._optimize_zeros(temp_path, output_path)
        
        # Удаляем временный файл
        try:
            os.remove(temp_path)
        except OSError:
            pass

        # Выводим статистику
        orig_size = os.path.getsize(input_path)
        comp_size = os.path.getsize(output_path)
        ratio = (1 - comp_size/orig_size) * 100
        print(f"Сжатие завершено. Коэффициент: {ratio:.2f}%")

    def _basic_compress(self, input_path: str, output_path: str):
        """Базовая компрессия без постобработки"""
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            # Заголовок (сигнатура 'NG', размер словаря)
            f_out.write(b'NG')
            f_out.write(len(self.ngram_dict).to_bytes(2, 'big'))
            
            buffer = bytearray()
            history = bytearray()
            
            while True:
                chunk = f_in.read(8192)
                if not chunk:
                    break
                buffer.extend(chunk)
                
                while len(buffer) >= self.min_match:
                    # 1. Проверка на повторяющиеся байты (RLE)
                    repeat_info = self.find_repeats(buffer)
                    if repeat_info and repeat_info[1] >= self.min_repeat_length:
                        byte, count = repeat_info
                        if byte == 0:  # Специальная обработка нулей
                            f_out.write(self.MARKER_ZEROS.to_bytes(2, 'big'))
                        else:  # Обработка других повторяющихся байтов
                            f_out.write(self.MARKER_REPEAT_BYTE.to_bytes(2, 'big'))
                            f_out.write(bytes([byte]))
                        
                        # Записываем длину последовательности
                        count = min(count, self.max_repeat_length)
                        f_out.write(count.to_bytes(2, 'big'))
                        
                        # Обновляем историю и буфер
                        history.extend(bytes([byte]) * count)
                        del buffer[:count]
                        continue
                    
                    # 2. Пробуем 8-грамму
                    if len(buffer) >= self.ngram_size:
                        ngram = bytes(buffer[:self.ngram_size])
                        if ngram in self.ngram_dict:
                            code = self.ngram_dict[ngram]
                            f_out.write(code.to_bytes(2, 'big'))
                            history.extend(buffer[:self.ngram_size])
                            del buffer[:self.ngram_size]
                            continue
                    
                    # 3. Пробуем LZ77 совпадение
                    match = self.find_best_match(history, buffer)
                    if match and match[1] >= self.min_match:
                        offset, length = match
                        f_out.write(self.MARKER_LZ77.to_bytes(2, 'big'))
                        f_out.write(offset.to_bytes(2, 'big'))
                        f_out.write(length.to_bytes(1, 'big'))
                        
                        # Копируем совпадающие данные в историю
                        history.extend(buffer[:length])
                        del buffer[:length]
                    else:
                        # 4. Сырой байт
                        f_out.write(self.MARKER_RAW_BYTE.to_bytes(2, 'big'))
                        f_out.write(buffer[:1])
                        history.append(buffer[0])
                        del buffer[:1]
                    
                    # Поддерживаем размер окна
                    if len(history) > self.window_size:
                        del history[:len(history)-self.window_size]
            
            # Остаток
            while buffer:
                f_out.write(self.MARKER_RAW_BYTE.to_bytes(2, 'big'))
                f_out.write(buffer[:1])
                del buffer[:1]

    def _optimize_zeros(self, input_path: str, output_path: str):
        """Постобработка: оптимизация последовательностей нулей"""
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            buffer = bytearray(f_in.read())
            i = 0
            n = len(buffer)
            
            # Копируем заголовок (первые 4 байта)
            if n >= 4:
                f_out.write(buffer[:4])
                i = 4
            
            while i < n:
                # Если текущая позиция - начало маркера (2 байта)
                if i + 1 < n:
                    marker = (buffer[i] << 8) | buffer[i+1]
                    
                    # Оптимизация последовательных RAW_BYTE с нулями
                    if marker == self.MARKER_RAW_BYTE and i + 2 < n and buffer[i+2] == 0:
                        # Начинаем подсчет нулей
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
                            # Заменяем на оптимизированную последовательность
                            f_out.write(self.MARKER_OPTIMIZED_ZEROS.to_bytes(2, 'big'))
                            f_out.write(zero_count.to_bytes(2, 'big'))
                            i = j
                            continue
                    
                    # Оптимизация последовательных MARKER_ZEROS
                    elif marker == self.MARKER_ZEROS and i + 4 < n:
                        length = (buffer[i+2] << 8) | buffer[i+3]
                        if length >= self.zero_optimization_threshold * 2:
                            # Объединяем с последующими последовательностями нулей
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
                            
                            if total_length > length:  # Нашли что объединять
                                f_out.write(self.MARKER_OPTIMIZED_ZEROS.to_bytes(2, 'big'))
                                f_out.write(total_length.to_bytes(2, 'big'))
                                i = j
                                continue
                
                # Если не нашли оптимизируемую последовательность, копируем как есть
                if i < n:
                    f_out.write(buffer[i:i+1])
                    i += 1

    def find_repeats(self, data: bytearray) -> Optional[Tuple[int, int]]:
        """Находит последовательность повторяющихся байтов"""
        if len(data) < self.min_repeat_length:
            return None
        
        first_byte = data[0]
        count = 1
        while count < len(data) and data[count] == first_byte:
            count += 1
            if count >= self.max_repeat_length:
                break
        
        return (first_byte, count) if count >= self.min_repeat_length else None

    def find_best_match(self, history: bytearray, buffer: bytearray) -> Optional[Tuple[int, int]]:
        """Поиск наилучшего совпадения для LZ77"""
        if not history or len(buffer) < self.min_match:
            return None
            
        best_offset, best_length = 0, self.min_match - 1
        max_possible = min(self.max_match, len(buffer))
        
        for offset in range(1, min(len(history), self.window_size) + 1):
            current_len = 0
            while (current_len < max_possible and 
                   offset + current_len - 1 < len(history) and 
                   current_len < len(buffer) and 
                   history[-offset + current_len] == buffer[current_len]):
                current_len += 1
            
            if current_len > best_length:
                best_length = current_len
                best_offset = offset
                if best_length >= self.max_match:
                    break
        
        return (best_offset, best_length) if best_length >= self.min_match else None

    def decompress(self, input_path: str, output_path: str):
        """Распаковка с поддержкой всех маркеров"""
        with open(input_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            # Проверка заголовка
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
                
                if marker_int == self.MARKER_RAW_BYTE:  # Сырой байт
                    byte = f_in.read(1)
                    if not byte:
                        break
                    output_buffer.append(byte[0])
                    f_out.write(byte)
                
                elif marker_int == self.MARKER_LZ77:  # LZ77
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
                    
                    for i in range(length):
                        pos = len(output_buffer) - offset
                        if 0 <= pos < len(output_buffer):
                            byte = output_buffer[pos]
                        else:
                            byte = 0
                        output_buffer.append(byte)
                        f_out.write(bytes([byte]))
                
                elif marker_int == self.MARKER_ZEROS:  # Последовательность нулей
                    length_bytes = f_in.read(2)
                    if not length_bytes:
                        raise ValueError("Неожиданный конец файла")
                    length = int.from_bytes(length_bytes, 'big')
                    output_buffer.extend(bytes(length))
                    f_out.write(bytes(length))
                
                elif marker_int == self.MARKER_REPEAT_BYTE:  # Повторяющийся байт
                    byte_data = f_in.read(1)
                    length_bytes = f_in.read(2)
                    if not byte_data or not length_bytes:
                        raise ValueError("Неожиданный конец файла")
                    
                    byte = byte_data[0]
                    length = int.from_bytes(length_bytes, 'big')
                    output_buffer.extend(bytes([byte]) * length)
                    f_out.write(bytes([byte]) * length)
                
                elif marker_int == self.MARKER_OPTIMIZED_ZEROS:  # Оптимизированные нули
                    length_bytes = f_in.read(2)
                    if not length_bytes:
                        raise ValueError("Неожиданный конец файла")
                    length = int.from_bytes(length_bytes, 'big')
                    output_buffer.extend(bytes(length))
                    f_out.write(bytes(length))
                
                else:  # 8-грамма
                    if marker_int < dict_size and marker_int in self.inverse_dict:
                        ngram = self.inverse_dict[marker_int]
                        output_buffer.extend(ngram)
                        f_out.write(ngram)
                    else:
                        output_buffer.extend(b'\x00' * self.ngram_size)
                        f_out.write(b'\x00' * self.ngram_size)

def load_ngrams(file_path: str) -> List[Tuple[bytes, float]]:
    """Загрузка 8-грамм из файла"""
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
    parser = argparse.ArgumentParser(description='Улучшенный компрессор ELF с оптимизацией нулей')
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
            
            compressor.compress(args.input, args.output)
            
        elif args.command == 'decompress':
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Файл не найден: {args.input}")
            
            compressor.decompress(args.input, args.output)
            print(f"Распаковка завершена. Результат: {args.output}")
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
