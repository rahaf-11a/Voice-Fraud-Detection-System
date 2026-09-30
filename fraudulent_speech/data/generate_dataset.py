# -*- coding: utf-8 -*-
"""
Fraud / Normal Saudi Arabic Dataset Generator

Clean GitHub version of the original Colab notebook:
Fraud_normal_dataset.ipynb

This script:
1. Generates a balanced 24,000-sentence dataset (12k normal / 12k fraud).
2. Generates a new balanced 100,000-sentence dataset with no overlap
   with the generated 24k dataset.

No Google Drive or personal Colab paths are used.
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


SUSPECT_LEXICON = {
    "بنك", "مصرف", "حسابك", "المحفظة", "شريحتك", "SIM", "الشريحة",
    "تم ايقاف", "إيقاف", "تم تعليق", "تعليق", "تجميد",
    "تحديث البيانات", "تأكيد الهوية", "توثيق", "تفويض", "التحقق",
    "رمز التحقق", "OTP", "الكود", "كود", "أرسل الرقم", "ابعث الرقم",
    "رابط", "اضغط الرابط", "حدث بياناتك", "بياناتك", "صورة الهوية", "هوية",
    "حول", "تحويل", "ايبان", "IBAN", "بطاقتك", "البطاقة", "مدى", "فيزا", "ماستر",
    "سداد", "فاتورة", "فواتير", "غرامة", "مخالفة", "نيابة", "شرطة", "محكمة",
    "جائزة", "رابح", "فزت", "استرداد", "استثمار", "عائد", "أرباح", "تداول", "كاش باك",
    "شحنة", "الجمارك", "رسوم", "تحرير الشحنة", "تأمين", "رسوم إضافية",
    "تم اختراق", "اختراق", "مصادقة", "3D", "ثري دي سيكيور", "secure",
    "اشترك", "اشتراك اجباري", "تجديد", "خصم تلقائي", "CODE"
}


def normalize_ar(text: str) -> str:
    t = text
    t = re.sub(r"[ـ]+", "", t)
    t = re.sub(r"[ًٌٍَُِّْٰ]", "", t)
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    t = t.replace("ى", "ي").replace("ة", "ه")
    t = t.replace("ؤ", "و").replace("ئ", "ي")
    t = re.sub(r"\s+", " ", t).strip()
    return t


WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u0600-\u06FF]+")


def word_tokenize(text: str):
    return WORD_RE.findall(text)


def extract_red_flags(text: str):
    norm = normalize_ar(text).lower()
    words = set(word_tokenize(norm))
    flags = set()

    for key in SUSPECT_LEXICON:
        k_norm = normalize_ar(key).lower()
        if " " in k_norm:
            if k_norm in norm:
                flags.add(key)
        else:
            if k_norm in words:
                flags.add(key)

    return sorted(flags)


greets = [
    "السلام عليكم", "هلا", "مرحبا", "يا مرحبا", "يا هلا", "السلام عليكم ورحمة الله",
    "هلابش", "هلابتس", "هلابك", "الو", "مرحبا بش", "مرحبا بتس",
    "يا هلا والله", "مسّاك الله بالخير", "صبّحك الله بالخير", "يا حيا الله", "عوداً حميداً",
    "يا مليون هلا", "حي الله من جانا", "الله يحييك", "ياهلا وغلا"
]

checks = [
    "وينك الحين؟", "متى ترجع؟", "وش وضعك؟", "فاضي دقيقة؟", "تقدر ترد؟", "متى تكون متاح؟",
    "وينتس؟", "وينش", "وينك", "وين الناس؟", "فاضيه؟", "متى ترجعين؟", "تقدرين تكلميني؟",
    "فاضيه لي؟", "متى فاضيه؟", "متى فاضي؟", "وش وضعتس؟", "وش وضعش؟",
    "ليه ما رديت؟", "لك ساعة ما ترد", "وصلتي ولا بعد؟", "طمنتينا؟", "أنت قريب؟",
    "وين موقف؟", "وين بتكون بعد شوي؟", "تردين علي إذا خلصتي؟"
]

ack = [
    "تمام", "اوكي", "تم", "خلاص", "طيب", "ابشر", "سم", "سمي", "اوك", "ابشري", "من عيوني",
    "على راسي", "متفقين", "علومك تمام", "ضبطنا", "توكلنا على الله", "تمّت",
    "تم يا بعدي", "كفو", "عاش"
]

asks_time = [
    "متى تفضى؟", "كم يبي له؟", "اليوم ولا بكره؟", "بعد المغرب ينفع؟", "قبل الدوام؟", "بعد الدوام؟",
    "بعد المغرب فاضي؟", "باتسر؟", "متى تفضين؟", "كم يبي لك؟", "بعد الظهر؟",
    "بعد العشاء؟", "الويكند مناسب؟", "اليوم العصر؟", "على الساعه تسع؟", "متى يناسبك؟",
    "تبين نأجله؟", "نقدم الموعد؟", "قبل الفجر بدري؟", "عقب الصلاة؟"
]

locations = [
    "البيت", "الدوام", "الجامعة", "المول", "المركز", "المستشفى", "الورشة", "الديوانيه", "المدرسه",
    "سوق الخضره", "المحطه", "البر", "الاستراحه", "البقالة", "المطار", "الصالة", "الحديقة",
    "المكتبة", "السوبرماركت", "المقهى", "الطريق الدائري", "المواقف", "الحي", "المستودع",
    "السوق الشعبي", "مغسلة السيارات", "محل الجوالات", "مخبز الحارة"
]

verbs = [
    "تقدر", "ممكن", "ودك", "يصلح", "يصير", "ينفع", "يمديك", "يمديش", "يمديتس", "تقدرين", "ودتس", "ودش",
    "توافق", "ترى", "تحب", "تبي", "تميل", "تقدرين تمرين", "تقدر تمر", "تسوّي", "تخلّي", "تأجل",
    "تفزع", "تتأكد", "تعلمني", "تطمنّي"
]

actions = [
    "تمر تاخذ الطلب", "تشيك على الموعد", "تسأل عن السعر", "تحجز لنا", "تتواصل معهم", "تغير الوقت",
    "ترسل الموقع", "تجيب خبز", "تشيك على السيارة", "تستعلم عن الفاتورة", "ترفع الصوت", "تقصّر الصوت",
    "تطلع برا", "تسافر معنا", "توقف عند البقالة", "تجيب قهوة معك", "تطمن علينا", "تفتح البوابة",
    "تشيّك على الطقس", "تصور الفاتورة وترسلها", "تذكّرني بكرة", "تشيّك على الطلبية", "تنسق مع السواق",
    "تجيب معك موّي", "تشيك على الإيميل", "تحدّثنا إذا تغير شي", "تستأذن من الشغل بدري",
    "تسجلنا في المكان", "تستفسر عن المواعيد", "تبدّل الموعد", "تجهز الأغراض"
]

resto = [
    "أبغى أطلب وجبه", "اطلب بيتزا وسط", "ضيف صوص زيادة", "بدّل المشروب", "كم وقت التوصيل؟",
    "فيه عرض اليوم؟", "برجر دبل", "جبن زياده", "لا مايونيز", "خليه حار خفيف", "ثنتين شاورما",
    "حط بطاطس كبير", "نبي عشاء خفيف", "بدون بصل لو سمحت", "زود خبز", "صلصة زيادة",
    "شيّك اذا عندهم رز اليوم", "خذ لنا مرق وقيمه", "لا تنسى الملعقة البلاستيك"
]

home = [
    "وصلت؟", "باقي كثير؟", "تبيني اجهز القهوة؟", "الماء بارد؟", "قفل النور اذا طلعت",
    "جيب الاغراض من الصيدلية", "جيب عشاء", "لا تتأخر", "نظّم الغرفة شوي", "مرّ على أخوك",
    "حط الغسيل في السلة", "إذا جيت دق الباب", "خذ المفتاح معك", "انتبه للدرج", "رّتب السفرة",
    "طفّي المكيف إذا مشيت", "جيب حليب", "جيب خبز توّك", "حط النفايات برا",
    "حط القدر على النار", "غسلت الفواكه؟", "نظفت المطبخ؟"
]

work = [
    "ارسل الملف بالإيميل", "موعد الاجتماع متى؟", "سلّم التقرير اليوم", "احتاج موافقة",
    "شيّك على التاسك", "قدّم الطلب في النظام", "حدّث التذكرة", "افتح النقاش مع الفريق",
    "ارفع المذكرة", "ثبت التغييرات", "ارسل رابط الاجتماع", "عطني آخر تحديث",
    "وش مخرجات الاجتماع؟", "بنحتاج عرض مختصر", "ارسال النسخة النهائية", "سلّم قبل نهاية اليوم",
    "اتفقنا على الخطة", "شيّك على الملاحظات", "قفل النقطة ذيك", "ركّبنا النسخة الجديدة"
]

study = [
    "جاهزه للمذاكره؟", "نراجع بعد العشاء؟", "ارسل الملخص", "كم بقى على الاختبار؟",
    "مشروعنا وين وصل؟", "سجّلت بالمقرر؟", "سلمت الواجب؟", "نلتقي بالمكتبة؟",
    "شرحتِ الدرس؟", "وش توقعات الاسئلة؟", "سويتي الملاحظات؟", "فتحتي منصة التعلم؟",
    "نقفل الكام ولا نخليها؟", "قسمتي الأدوار؟", "ارسل رابط المصادر", "قدّمتي العرض؟",
    "درستي المحاضرة المسجلة؟", "المعيد رد؟", "طلعت الدرجات؟"
]

smalltalk = [
    "ترى الجو اليوم حلو مره", "الزحمة كانت ما تنطاق", "وصلني الطلب متأخر شوي",
    "جربت القهوة الجديدة وكانت مزبوطة", "النت عندي يلعب اليوم", "البطارية بتخلص",
    "توني طالع من الدوام", "اجتمعت معهم وكانو طيبين", "المطر كان قوي بالحارة",
    "انقطعت الكهرب أمس شوي", "الحر شديد هاليومين", "برد الصبح فلة", "أحسني تعبان شوي",
    "نمت متأخر البارح", "صحيت بدري اليوم", "ترى العرض بالمحل كان رهيب",
    "السيارة سوت صوت غريب", "الطريق الثاني أسرع", "خلص الخبز من البيت",
    "بديت أمشي على الدايت", "بأطلع أمشي بعد المغرب", "اشتريت كفر للجوال",
    "العيال اليوم نشيطين", "البيت يبيله ترتيب", "بنسافر نهاية الأسبوع",
    "قريت خبر يضحك", "المسلسل اللي قلتِ لي عنه رهيب", "الكتاب اللي عندي وصل",
    "جربت تطبيق مفيد", "العامل جا وخلص الشغل", "فاتتني صلاة الجماعة اليوم",
    "بنغير الكنبة قريب", "مرّيت على البقالة وكان زحمة"
]

reactions = [
    "يا ساتر!", "يا لطيف", "الله يستر", "ما شاء الله", "ههههه وربي ضحكت", "يا سلام",
    "والله خوش خبر", "ايه والله صدقت", "عاد برافو عليهم", "لا حول ولا قوة الا بالله",
    "الله يعين", "الله يكتب لنا اللي فيه الخير", "كذا تمام", "كذا مضبوط", "حماس",
    "يازين هالخبر", "ياخي رهيب", "قسم إنك فنان", "ما قصرتي", "جزاك الله خير"
]

closings = [
    "يلا أشوفك على خير", "مع السلامة", "نرجع نتواصل بعدين", "أكلمك إذا خلصت",
    "خلي اتصالنا بالليل", "برجع لك بعد شوي", "توصلين بالسلامة", "نخليها لبكرة",
    "يعطيك العافية", "الله يحفظك", "لا تطولين علينا", "سلام"
]

particles = [
    "لو سمحت", "تكفى", "يا اخوي", "يا اختي", "من فضلك", "بس", "عافيّه", "الله يقويك",
    "الله يرضى عليك", "لا هنت", "لا تهون", "الله يحييك", "يا بعدي", "يا الغلا", "يا قلبي"
]

time_refs = [
    "الحين", "اليوم", "بكره", "بعد شوي", "بعد العصر", "بعد المغرب", "بالليل", "الويكند",
    "الصبح", "الضحى", "قبل الظهر", "قبل المغرب", "على طول", "هالحزة"
]

fraud_openers = [
    "معك خدمة العملاء", "قسم التحقق", "مركز الدعم", "مكالمه مسجله", "جهة رسمية", "التواصل العاجل",
    "قسم الأمن السيبراني", "إدارة الحسابات", "الدعم الفني", "مركز شكاوى العملاء", "هيئة رسمية",
    "البنك المركزي", "قسم حماية المستهلك", "خدمة التوثيق", "المحكمة الالكترونية"
]

fraud_actions = [
    "تم ايقاف حسابك مؤقتاً", "نحتاج تحديث البيانات فوراً", "راح نرسل لك رمز التحقق الآن",
    "لا تشارك الكود مع احد", "لتفعيل شريحتك لازم توثيق", "رابط التحديث بيجيك برساله",
    "عندك رسوم على الشحنة بالجمارك", "فزت بجائزة خاصة", "في خصم غير مصرح به على بطاقتك",
    "نحتاج الايبان للتحويل", "لرفع ايقاف الحساب اضغط الرابط", "تم تعليق محفظتك",
    "عندك مخالفة ولم تسدد الغرامه", "نطالب بتأكيد الهوية",
    "تم تجميد بطاقتك الائتمانية", "المحفظة الالكترونية موقوفة", "يجب تسديد الفاتورة الآن",
    "تم تسجيل دخول مشبوه", "مطلوب إعادة ضبط كلمة المرور", "يجب دفع رسوم التأمين", "تم رصد اختراق"
]

fraud_urgency = [
    "بشكل عاجل", "خلال عشر دقايق", "الآن", "فوراً", "اليوم", "قبل ما ينقفل الحساب", "الحين",
    "في اقرب وقت", "قبل نهاية اليوم", "بلا تأخير", "ضروري جداً", "مستعجل", "هالحظة", "بأسرع وقت"
]

fraud_follow = [
    "ارسل الكود هنا", "ابعث رقم الهويه", "عطني الايبان", "اضغط على الرابط للتحقق",
    "لا تكفل المكالمه", "لا تبلغ احد", "خلك معي لين نكمل الاجراء",
    "صور بطاقتك وارسلها", "ادخل الرقم في الرسالة", "اعطنا رمز التفعيل",
    "حدث بياناتك عبر الرابط", "لا تخبر أحد بالبيانات", "رد برسالة تأكيد",
    "شارك الرقم معنا", "اكتب الكود الحين", "ارسل صورة الهوية الوطنية"
]

dialect_particles = [
    "يا الغالي", "يا اختي", "يا اخوي", "تكفى", "لو سمحت", "بس", "عاجل", "لا تهمل", "خذ الموضوع جد",
    "انتبه", "لا تتأخر", "ركز معي", "رجاءً", "بسرعة", "من فضلك"
]


def make_normal_sentence():
    pattern = random.choice([
        "{greet} {check}",
        "{verb} {action}؟",
        "{ask_time}",
        "وينك عن {loc}؟",
        "{resto}",
        "{home}",
        "{work}",
        "{study}",
        "{ack} نلتقي {time_ref}؟",
        "{small}",
        "{small} {react}",
        "{small} {close}",
        "{greet} {small}",
        "{greet} {ack}",
        "{small} وبعدها إن شاء الله {time_ref}",
        "ترى {small}".replace("ترى ترى", "ترى "),
    ])

    base = pattern.format(
        greet=random.choice(greets),
        check=random.choice(checks),
        verb=random.choice(verbs),
        action=random.choice(actions),
        ask_time=random.choice(asks_time),
        loc=random.choice(locations),
        resto=random.choice(resto),
        home=random.choice(home),
        work=random.choice(work),
        study=random.choice(study),
        ack=random.choice(ack),
        time_ref=random.choice(time_refs),
        small=random.choice(smalltalk),
        react=random.choice(reactions),
        close=random.choice(closings),
    )

    if random.random() < 0.45:
        base = f"{random.choice(particles)}، {base}"

    return base if not extract_red_flags(base) else None


def make_fraud_sentence():
    pattern = random.choice([
        "{opener} {action} {urgency} {follow}",
        "{opener} {action} {follow} {urgency}",
        "{opener} {action} {urgency} {follow}",
    ])

    sent = pattern.format(
        opener=random.choice(fraud_openers),
        action=random.choice(fraud_actions),
        urgency=random.choice(fraud_urgency),
        follow=random.choice(fraud_follow),
    )

    if random.random() < 0.4:
        sent = f"{random.choice(dialect_particles)}، {sent}"

    return sent


def to_record(sentence: str):
    words = word_tokenize(sentence)
    flags = extract_red_flags(sentence)
    label = "fraud" if flags else "normal"

    return {
        "speech": sentence,
        "words_list": words,
        "red_flags": flags,
        "type": label,
    }


def generate_balanced_dataset(target_size: int, existing_speech=None, seed: int = 777):
    random.seed(seed)
    half = target_size // 2
    max_tries = half * 100
    existing_speech = existing_speech or set()

    normal_set, fraud_set = set(), set()

    pbar_normal = tqdm(total=half, desc="Generating NORMAL", unit="sent")
    tries = 0
    while len(normal_set) < half and tries < max_tries:
        sentence = make_normal_sentence()
        tries += 1
        if sentence and sentence not in normal_set and sentence not in existing_speech and not extract_red_flags(sentence):
            normal_set.add(sentence)
            pbar_normal.update(1)
    pbar_normal.close()

    pbar_fraud = tqdm(total=half, desc="Generating FRAUD", unit="sent")
    tries = 0
    while len(fraud_set) < half and tries < max_tries:
        sentence = make_fraud_sentence()
        tries += 1
        if sentence and sentence not in fraud_set and sentence not in existing_speech and extract_red_flags(sentence):
            fraud_set.add(sentence)
            pbar_fraud.update(1)
    pbar_fraud.close()

    if len(normal_set) < half or len(fraud_set) < half:
        raise RuntimeError("Target size was not reached. Increase MAX_TRIES or expand the templates.")

    records = [to_record(s) for s in normal_set] + [to_record(s) for s in fraud_set]
    df = pd.DataFrame(records, columns=["speech", "words_list", "red_flags", "type"])

    if existing_speech:
        overlap = existing_speech.intersection(set(df["speech"]))
        assert not overlap, f"Overlap with existing dataset: {len(overlap)} sentences"

    assert df.duplicated(subset=["speech"]).sum() == 0, "Duplicate sentences found."
    return df


def save_dataset(df: pd.DataFrame, csv_path: Path, xlsx_path: Path):
    save_df = df.copy()
    save_df["words_list"] = save_df["words_list"].apply(lambda x: json.dumps(x, ensure_ascii=False))
    save_df["red_flags"] = save_df["red_flags"].apply(lambda x: json.dumps(x, ensure_ascii=False))

    save_df.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    print(f"Saved CSV: {csv_path}")

    try:
        with pd.ExcelWriter(xlsx_path) as writer:
            save_df.to_excel(writer, index=False)
        print(f"Saved XLSX: {xlsx_path}")
    except Exception as exc:
        print(f"XLSX save failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Generate balanced Saudi Arabic fraud/normal datasets.")
    parser.add_argument(
