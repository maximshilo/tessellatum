# Bundled detection models

The face detection in `tessellatum.core.faces` runs on the first two of these files,
the subject detection in `tessellatum.core.subject` on the third, and the text
detection in `tessellatum.core.text` on the fourth. They ship with the app and are
read offline. The face models are MIT-licensed; U2-Net-p is under the Apache License
2.0, as this repository is (its text is the repository's `LICENSE`), in the ONNX form
rembg (MIT) converted it to; PP-OCRv6-small is under the Apache License 2.0 too. The
notices are below.

| file | what it is | source | license | SHA-256 |
|---|---|---|---|---|
| `face_detection_yunet_2026may.onnx` | YuNet, a small face detection network (232 KB), run by OpenCV's DNN module | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet), re-exported there from [libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train) with dynamic input dimensions | MIT, Copyright (c) 2020 Shiqi Yu | `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0` |
| `lbpcascade_animeface.xml` | lbpcascade_animeface, a cascade of LBP features for anime and manga faces (247 KB) | [nagadomi/lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) | MIT, Copyright (C) 2011 by nagadomi@nurs.or.jp (the notice is also at the top of the file) | `9376d30ac38db6bda2a68b88b3b76bbd7e6aa33af47f7f5c76bc88ca75f1ce30` |
| `u2netp.onnx` | U2-Net-p, a small salient object detection network (4.6 MB), run by OpenCV's DNN module | the `u2netp` weights of [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) (Qin et al., "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection", Pattern Recognition 2020), as [rembg](https://github.com/danielgatis/rembg) converts them to ONNX (release `v0.0.0`, file `u2netp.onnx`) | Apache License 2.0 (U-2-Net); the conversion MIT, Copyright (c) 2020 Daniel Gatis | `309c8469258dda742793dce0ebea8e6dd393174f89934733ecc8b14c76f4ddd8` |
| `PP-OCRv6_small_det.onnx` | PP-OCRv6-small's text detection network (9.9 MB), run by ONNX Runtime | PaddlePaddle's own ONNX export of [PP-OCRv6_small_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det), from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR): the file `inference.onnx` of [PaddlePaddle/PP-OCRv6_small_det_onnx](https://huggingface.co/PaddlePaddle/PP-OCRv6_small_det_onnx) | Apache License 2.0 (PaddleOCR's `LICENSE` and the model's card) | `d73e0058b7a8086bbd57f3d10b8bcd4ff95363f67e06e2762b5e814fe9c9410e` |

The SHA-256 of the XML file is of its bytes as downloaded, with LF line endings.

## YuNet

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## lbpcascade_animeface

```
The MIT License (MIT)

Copyright (C) 2011 by nagadomi@nurs.or.jp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## U2-Net-p

U-2-Net is licensed under the Apache License, Version 2.0, whose full text is this
repository's own `LICENSE`. Its repository adds no NOTICE file and no copyright line
of its own. The ONNX file is rembg's conversion of the published weights, under rembg's
license:

```
MIT License

Copyright (c) 2020 Daniel Gatis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## PP-OCRv6-small

PP-OCRv6 is licensed under the Apache License, Version 2.0, whose full text is this
repository's own `LICENSE`, as its model card and PaddleOCR's repository state.
PaddleOCR's repository adds no NOTICE file, and its `LICENSE` no copyright line of
its own. The file is the model's authors' own ONNX export; it computes the same map as
the conversion RapidOCR ships in its wheel.
