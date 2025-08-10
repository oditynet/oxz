# oxz

# Сжатие завершено. Коэффициент: 62.03%

Алгоритм сжатия построен на обучении по N-gramm.

Пока идут тестирования и обучение машины, но вы только представьте...
Вы берете свою рабочую машину и на ELF файлах начинаете сами обучать систему. Вы можете обучить ее ена малеькие файлы или на большие.
Потом вы берете копрессор и сжимаете все свои файлы. Средняя компрессия на 8-gramm выходила до 62%. Пока идут тесты с большим количеством файлов и другими N-gramm  для достижения лучшего сжатия.

Работает все в один поток.

```
find /home/odity/bin -type f -size -20k -perm -u+x -exec sh -c 'cp -v "$1" /tmp/' sh {} \;
python ngram.py /tmp/ 8gramm --max 8
python oxz.py compress <file in>  <file out> 8gramm

python oxz.py decompress <file in> <file out> 8gramm
```

<img src="https://github.com/oditynet/oxz/blob/main/result.png" title="example" width="800" />

TODO: 1) Скорость сжатия. Это слезы. до 100kb еще жить можно.
