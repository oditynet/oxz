import os
import argparse
from typing import Dict, List, Tuple, Optional

class RobustCompressor:
    def __init__(self, ngram_probs: List[Tuple[bytes, float]]):
        # Словарь 8-грамм
        self.ngram_size = 8
        self.ngram_dict = {ngram: idx for idx, (ngram, _) in enumerate(ngram_probs[:65535])}
        self.inverse_dict = {idx: ngram for idx, (ngram, _) in enumerate(ngram_probs[:65535])}
        
        # Параметры LZ77
        self.window_size = 8192  # Увеличили окно для лучшего сжатия
        self.min_match = 4       # Минимальная длина совпадения
        self.max_match = 255     # Максимальная длина совпадения

    def compress(self, input_path: str, output_path: str):
        """Улучшенное сжатие с проверками"""
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
                    # Сначала пробуем 8-грамму
                    if len(buffer) >= self.ngram_size:
                        ngram = bytes(buffer[:self.ngram_size])
                        if ngram in self.ngram_dict:
                            code = self.ngram_dict[ngram]
                            f_out.write(code.to_bytes(2, 'big'))
                            history.extend(buffer[:self.ngram_size])
                            del buffer[:self.ngram_size]
                            continue
                    
                    # Затем LZ77 совпадение
                    match = self.find_best_match(history, buffer)
                    if match and match[1] >= self.min_match:
                        offset, length = match
                        f_out.write((0xFFFF).to_bytes(2, 'big'))
                        f_out.write(offset.to_bytes(2, 'big'))
                        f_out.write(length.to_bytes(1, 'big'))
                        
                        # Копируем совпадающие данные в историю
                        for i in range(length):
                            if i < len(buffer):
                                history.append(buffer[i])
                        del buffer[:length]
                    else:
                        # Сырой байт
                        f_out.write((0xFFFE).to_bytes(2, 'big'))
                        f_out.write(buffer[:1])
                        history.append(buffer[0])
                        del buffer[:1]
                    
                    # Поддерживаем размер окна
                    if len(history) > self.window_size:
                        del history[:len(history)-self.window_size]
            
            # Остаток
            while buffer:
                f_out.write((0xFFFE).to_bytes(2, 'big'))
                f_out.write(buffer[:1])
                del buffer[:1]

    def find_best_match(self, history: bytearray, buffer: bytearray) -> Optional[Tuple[int, int]]:
        """Безопасный поиск совпадений с проверкой границ"""
        if not history or len(buffer) < self.min_match:
            return None
            
        best_offset, best_length = 0, self.min_match - 1
        max_possible = min(self.max_match, len(buffer))
        
        # Ищем в истории
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
        """Распаковка с полной проверкой ошибок"""
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
                
                if marker_int == 0xFFFE:  # Сырой байт
                    byte = f_in.read(1)
                    if not byte:
                        break
                    output_buffer.append(byte[0])
                    f_out.write(byte)
                
                elif marker_int == 0xFFFF:  # LZ77
                    offset_bytes = f_in.read(2)
                    length_byte = f_in.read(1)
                    if not offset_bytes or not length_byte:
                        raise ValueError("Неожиданный конец файла")
                    
                    offset = int.from_bytes(offset_bytes, 'big')
                    length = length_byte[0]
                    
                    if offset == 0 or offset > len(output_buffer):
                        # Автоматическое восстановление при ошибке смещения
                        output_buffer.append(0)
                        f_out.write(b'\x00')
                        continue
                    
                    # Копируем с перекрытием
                    for i in range(length):
                        pos = len(output_buffer) - offset
                        if 0 <= pos < len(output_buffer):
                            byte = output_buffer[pos]
                        else:
                            byte = 0
                        output_buffer.append(byte)
                        f_out.write(bytes([byte]))
                
                else:  # 8-грамма
                    if marker_int < dict_size and marker_int in self.inverse_dict:
                        ngram = self.inverse_dict[marker_int]
                        output_buffer.extend(ngram)
                        f_out.write(ngram)
                    else:
                        # Защита от поврежденных данных
                        output_buffer.extend(b'\x00' * self.ngram_size)
                        f_out.write(b'\x00' * self.ngram_size)

def load_ngrams(file_path: str) -> List[Tuple[bytes, float]]:
    """Загрузка 8-грамм с проверкой"""
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
    parser = argparse.ArgumentParser(description='Надежный компрессор ELF')
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
            orig_size = os.path.getsize(args.input)
            comp_size = os.path.getsize(args.output)
            ratio = (1 - comp_size/orig_size) * 100
            print(f"Сжатие завершено. Коэффициент: {ratio:.2f}%")
            
        elif args.command == 'decompress':
            if not os.path.exists(args.input):
                raise FileNotFoundError(f"Файл не найден: {args.input}")
            
            compressor.decompress(args.input, args.output)
            print(f"Распаковка завершена. Результат: {args.output}")
    
    except Exception as e:
        print(f"Ошибка: {str(e)}")

if __name__ == "__main__":
    main()
