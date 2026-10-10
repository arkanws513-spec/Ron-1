# Ron-1

رون-1 مساعد لغوي يعمل محليًا داخل المتصفح باستخدام نموذج **SmolLM2-135M-Instruct Q4F16**. مستودع GitHub هو المصدر الأساسي للكود والأوزان وملفات التشغيل.

## تجربة المحادثة

افتح [Ron-1 على GitHub Pages](https://arkanws513-spec.github.io/Ron-1/)، واضغط **تشغيل نواة SmolLM2**، ثم انتظر اكتمال التحميل. بعد ذلك اكتب رسالة واضغط إرسال.

## الاستضافة والتشغيل

- الواجهة وملف Worker: محفوظان في هذا المستودع.
- مكتبة Transformers.js وملفات ONNX Runtime WASM: تُحفظ في `vendor/transformers/` وتُقدَّم من GitHub Pages.
- إعدادات النموذج والمفردات: تُحفظ في `models/onnx-community/SmolLM2-135M-Instruct-ONNX/` وتُقدَّم من GitHub Pages.
- ملف الأوزان Q4 الكبير: مستضاف في [GitHub Releases](https://github.com/arkanws513-spec/Ron-1/releases/tag/ron1-smollm2-135m-q4-v1)، وحجمه نحو 117 ميجابايت.
- الاستدلال يتم داخل متصفح المستخدم؛ لا يوجد API مدفوع أو خادم استدلال خارجي في مسار المحادثة.

يُجهّز سير عمل GitHub Actions ملفات التشغيل والتهيئة الناقصة مرة واحدة ويحفظها في المستودع، ثم ينشر نسخة الموقع من ملفات GitHub. بعد اكتمال التجهيز، لا يحتاج تشغيل المحادثة إلى CDN أو تنزيل ملفات النموذج من منصات خارجية. تنزيل الأوزان لأول مرة ما زال مطلوبًا إذا لم تكن مخزنة مؤقتًا على الجهاز.

## حدود النموذج

SmolLM2-135M نموذج صغير جاهز، وليس نموذجًا دُرّب من الصفر داخل Ron-1. قدرته محدودة مقارنة بالنماذج الكبيرة، وقد تكون إجاباته أو استمرارية الحوار غير مثالية. نجاح النشر وحده لا يثبت عمله على كل جهاز؛ يلزم اختبار تحميله وتوليد رد فعلي.

## ملفات أساسية

- `index.html`: واجهة المحادثة.
- `ron-model-worker.js`: تحميل النموذج وتوليد الردود داخل Worker.
- `.github/workflows/pages.yml`: تجهيز الأصول المحلية ونشر الموقع.


## توافق الأجهزة محدودة الذاكرة

يستخدم Ron-1 الآن نسخة ONNX **Q4F16** الأصغر (نحو 117 MB بدلًا من نحو 181 MB لنسخة Q4)، بهدف خفض ضغط الذاكرة أثناء تهيئة جلسة ONNX على الهواتف القديمة. يجري تجهيز هذه النسخة في GitHub Actions والتحقق من SHA-256 قبل رفعها إلى GitHub Release؛ لا ينزّل المتصفح النموذج من Hugging Face وقت التشغيل.


The Q4F16 runtime configuration is versioned under `models-v2/` to avoid stale browser-cache metadata. Its KV-cache dtype is explicitly set to `float32`, matching the ONNX session input contract observed during browser inference tests.
