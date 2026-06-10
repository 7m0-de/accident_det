# AI-Powered Traffic Accident Detection System

An end-to-end, high-performance intelligent traffic accident and hazard detection pipeline. The system utilizes YOLOv10 for custom object detection, BoT-SORT for visual object tracking, and Bayesian Decision Fusion coupled with kinematic Time-to-Collision (TTC) models to verify collisions in real-time. It also features integrated fire and smoke detection.

---

## 🏗️ الهيكلية البرمجية الجديدة (New Modular Architecture)

تمت إعادة هيكلة المشروع وفصل المسؤوليات لتحسين صيانة الأكواد وكفاءتها. يوضح الهيكل التالي الملفات الأساسية ودور كل منها:

### 1. المكونات الخلفية (Backend Python Modules)
- **[server.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/server.py)**: خادم FastAPI المستضيف لـ REST APIs الخاصة برفع وتحليل الفيديو وجلب التقدم (Progress).
- **[logic.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/logic.py)**: واجهة استدعاء تصديرية مبسطة (Wrapper Entry Point) لتسهيل استدعاء النظام من خادم الويب دون كسر توافق الأكواد السابقة.
- **[pipeline.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/pipeline.py)**: المحرك الرئيسي لإدارة خيوط المعالجة المتعددة (Multithreading)، حيث يدير قراءة الإطارات وكتابة الفيديو النهائي واستدعاء النماذج.
- **[trackers.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/trackers.py)**: يحتوي على فئات التتبع البصري وحسابات التدفق الضوئي والتنبؤ الحركي (`TimeLocker`, `KinematicTrigger`, `GlobalMotionComp`, `KalmanTracker`, `OpticalFlow`, `MotionAnalyzer`).
- **[risk.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/risk.py)**: يحتوي على فئات تقييم المخاطر وتأكيد الاصطدام وحسابات الاندماج البيزي (`Registry`, `BayesianFusion`, `KinematicModel`, `RiskEngine`).
- **[fire.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/fire.py)**: يحتوي على كاشف النيران والدخان الديناميكي في مناطق الاهتمام (ROIs) وكامل الإطار (`FireDetector`).
- **[helpers.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/helpers.py)**: يحتوي على العمليات الحسابية المساعدة وفئة تهيئة الإعدادات المحلية `Config`.
- **[config.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/config.py)**: ملف الإعدادات المركزي لتهيئة متغيرات النظام وحساسيات المستشعرات الحركية ومسارات النماذج بشكل مباشر ودون تراجع تلقائي (No Fallbacks).
- **[utils.py](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/utils.py)**: يحتوي على دوال ترميز وتغيير تنسيق الفيديوهات لتناسب المتصفحات (FFmpeg) ودوال تنظيف الملفات المؤقتة.

### 2. الواجهة الأمامية (Frontend Dashboard)
- **[index.html](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/index.html)**: الهيكل الأساسي للوحة التحكم التفاعلية خفيفة الحجم (تم اختزالها من 170KB إلى 30KB).
- **[static/css/style.css](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/static/css/style.css)**: ملف التنسيقات والمؤثرات البصرية والألوان (CSS) المستخرج خارجياً.
- **[static/js/app.js](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/static/js/app.js)**: السكربت البرمجي المسؤول عن الاتصال بالخادم وتحديث الرسوم البيانية وتشغيل التنبيهات الصوتية.

---

## 🛠️ التثبيت والتشغيل (Installation & Setup)

### 1. تثبيت المكتبات المطلوبة (Install Dependencies)
تأكد من استخدام Python 3.9+، ثم قم بتثبيت المتطلبات يدوياً (تم تعطيل التثبيت التلقائي لتجنب التعليق وقت التشغيل):
```bash
pip install -r requirements.txt
```

> [!NOTE]
> إذا كنت تستخدم بطاقة رسوميات Nvidia وتريد تفعيل تسريع المعالجة بـ CUDA، يرجى تثبيت PyTorch يدوياً عبر:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### 2. التحقق من أداة FFmpeg
يجب أن تكون أداة `ffmpeg` مثبتة على نظام التشغيل ومضافة لمتغيرات البيئة (PATH) ليتمكن الخادم من تحويل صيغ الفيديوهات المنتجة إلى تنسيقات متوافقة للبث الفوري على المتصفح.

### 3. تشغيل الخادم الخلفي (Start backend server)
قم بتشغيل خادم FastAPI المحلي:
```bash
python server.py --host 0.0.0.0 --port 8000
```

### 4. الوصول إلى لوحة التحكم (Access dashboard)
يمكنك فتح لوحة التحكم مباشرة بفتح [index.html](file:///C:/Users/ayman/Documents/Ai/boss/FinalFream%20N/index.html) في أي متصفح ويب حديث، ثم إدخال رابط الاتصال بالخادم (مثال: `http://localhost:8000`) والضغط على **Establish Connection**.
