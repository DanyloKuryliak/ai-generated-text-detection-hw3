"""Build the measured final report, visual appendix, and results notebook."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle, FancyArrowPatch
import nbformat as nbf
import numpy as np

ROOT=Path(__file__).resolve().parent
INK="#172d3a"; MUTED="#52616a"
COLORS=["#386789","#c67c45","#518978"]
LABELS=["human","ai","fusion"]
NAMES={"word":"Слова TF-IDF","character":"Символи TF-IDF","classical_blend":"Слова + символи",
       "transformer_head":"DistilBERT: початок","transformer_chunks":"DistilBERT: 3 частини","ensemble":"Підсумковий ансамбль"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False,"text.color":INK,
                     "axes.labelcolor":INK,"axes.titleweight":"bold"})


def load(name):
    return json.loads((ROOT/"results"/name).read_text())


def t(fig,x,y,value,size=10,weight="normal",color=INK,**kw):
    return fig.text(x,y,value,va="top",fontsize=size,weight=weight,color=color,linespacing=1.4,**kw)


def heading(fig,y,num,value):
    t(fig,.065,y,f"{num:02}",12,"bold",COLORS[0]); t(fig,.12,y,value,12,"bold")


def footer(fig,page):
    fig.add_artist(plt.Line2D([.065,.935],[.044,.044],color="#d8e0e4",lw=.8))
    t(fig,.065,.034,"UCU AI · HW3 · Визначення походження тексту",8,color=MUTED)
    t(fig,.935,.034,f"{page} / 3",8,color=MUTED,ha="right")


def number(x,digits=4):
    return f"{x:.{digits}f}".replace(".",",")


def main():
    m=load("final_metrics.json"); audit=load("data_audit.json"); split=load("split_audit.json")
    config=load("transformer_config.json")
    classical=load("classical_metrics.json")
    char_weight=classical["models"]["classical_blend"]["character_weight"]
    token=load("development_tokenization.json")
    attribution=load("character_attributions.json")
    receipt_path=ROOT/"results/kaggle_submissions.json"
    receipts=json.loads(receipt_path.read_text()) if receipt_path.exists() else {"status":"not_submitted"}
    selected=m["selected_model"]; final=m["models"][selected]
    model_names=list(m["models"])
    delta_char=m["models"]["character"]["cv_mean"]-m["models"]["word"]["cv_mean"]
    delta_chunk=m["models"]["transformer_chunks"]["cv_mean"]-m["models"]["transformer_head"]["cv_mean"]
    hypotheses={"character_cv_delta":delta_char,"chunking_cv_delta":delta_chunk,
                "character_supported_in_this_configuration":delta_char>0,
                "chunking_supported_in_this_configuration":delta_chunk>0}
    (ROOT/"results/hypotheses.json").write_text(json.dumps(hypotheses,indent=2)+"\n")
    with PdfPages(ROOT/"FINAL_REPORT_UK.pdf") as pdf:
        pdf.infodict().update({"Title":"Визначення походження тексту: результати експериментів",
                               "Subject":"Двосторінковий звіт і візуальний додаток","Author":""})
        fig=plt.figure(figsize=(8.27,11.69),facecolor="white")
        t(fig,.065,.955,"Визначення походження тексту",20,"bold")
        t(fig,.065,.917,"Дані, рішення та перевірка двох гіпотез",11,color=MUTED)
        heading(fig,.865,1,"Дані та початкові спостереження")
        t(fig,.065,.835,"Задача: human — людина; ai — модель; fusion — частково переписаний текст.\n"
          "272 352 навчальні та 25 000 тестових документів англійською й німецькою.\n"
          "Пропусків, повторних ID й точних повторів після очищення пробілів не виявлено.")
        ax=fig.add_axes([.13,.575,.30,.18])
        values=np.array([audit["train"]["counts"]["label"][k] for k in LABELS])/audit["train"]["rows"]*100
        bars=ax.bar(LABELS,values,color=COLORS,width=.65)
        ax.set(ylim=(0,54),ylabel="Частка документів, %")
        ax.set_title("Нерівномірні класи",fontsize=10,pad=10)
        ax.bar_label(bars,labels=[number(x,1)+"%" for x in values],padding=4,fontsize=9)
        ax.grid(axis="y",alpha=.15); ax.set_axisbelow(True)
        ax=fig.add_axes([.64,.575,.29,.18])
        domains=["arxiv_papers","gutenberg","cnn_news","student_essays"]
        rows=np.array([[sum(x["count"] for x in audit["train"]["language_domain_label"] if x["domain"]==d and x["label"]==k) for k in LABELS] for d in domains],float)
        rows=rows/rows.sum(1,keepdims=True)*100; offset=np.zeros(4)
        for j,k in enumerate(LABELS):
            ax.barh(["Наукові статті","Книги","Новини","Есе"],rows[:,j],left=offset,color=COLORS[j],height=.6,label=k); offset+=rows[:,j]
        ax.invert_yaxis(); ax.set(xlim=(0,100),xlabel="Частка документів, %")
        ax.set_title("Джерело впливає на баланс",fontsize=10,pad=10)
        ax.legend(ncol=3,bbox_to_anchor=(.5,-.27),loc="lower center",frameon=False,fontsize=8,handlelength=1,columnspacing=.7)
        t(fig,.065,.515,"Медіанна довжина — 463 слова; 43,4% документів довші за 512 слів.\n"
          "Тому перевірено читання різних частин тексту; слова не дорівнюють токенам.")
        heading(fig,.450,2,"Розбиття та захист від витоку")
        t(fig,.065,.420,f"MinHash LSH і точна схожість фрагментів із трьох слів (Жаккар ≥ 0,65)\n"
          f"виявили {split['verified_pairs']:,} близьких пар. Виявлені групи не перетинаються між частинами.\n".replace(","," ")+
          "40 000 документів — навчання; 54 469 — фіксований контроль. Seed: 42.\n"
          "На тих самих трьох групових розбиттях обирали моделі та ваги ансамблю.\n"
          "Контроль не використовувався для вибору ваг. Групування може пропускати перефразування.",9.6)
        heading(fig,.307,3,"Моделі та причини вибору")
        t(fig,.065,.277,
          "База: TF-IDF слів і пар слів (80 000 ознак) + логістична регресія.\n"
          "Малий пошук C=1/4 обрав C=4. Словник навчався лише всередині навчальних розбиттів.\n"
          "Гіпотеза 1: символьні фрагменти 3–5 знаків (100 000 ознак, C=4)\n"
          "можуть уловлювати дрібні лексичні, пунктуаційні та стилістичні відмінності.\n"
          "Енкодер: багатомовний DistilBERT; 1 епоха, 256 токенів, LR=3×10⁻⁵,\n"
          f"batch=16, накопичення градієнтів=2, FP16; {config['gpu']}.\n"
          "Гіпотеза 2: ті самі ваги моделі; початок тексту порівняно з усередненням\n"
          "ймовірностей для початку, середини й кінця — до 254 змістових токенів на частину.\n"
          "Додатково перевірено суміш ймовірностей моделей із вагами 0/0,25/0,5/0,75/1.",9.6)
        t(fig,.065,.081,"Усі моделі навчалися на тих самих ID. Тестові мітки не використовувалися.\n"
          "Використано лише train.csv і дозволені попередньо навчені ваги.",9,color=MUTED)
        footer(fig,1); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(8.27,11.69),facecolor="white")
        t(fig,.065,.955,"Результати та висновки",20,"bold")
        t(fig,.065,.917,"Оцінки кросвалідації та контролю розділено",11,color=MUTED)
        heading(fig,.865,4,"Порівняння на однакових документах")
        ax=fig.add_axes([.32,.605,.59,.205])
        y=np.arange(len(model_names))
        cv=np.array([m["models"][n]["cv_mean"] for n in model_names])
        std=np.array([m["models"][n]["cv_std"] for n in model_names])
        held=np.array([m["models"][n]["holdout_macro_f1"] for n in model_names])
        ax.errorbar(cv,y-.09,xerr=std,fmt="o",color=COLORS[0],capsize=3,label="CV: середнє ± SD")
        ax.scatter(held,y+.09,color=COLORS[2],marker="s",s=27,label="Окремий контроль")
        ax.set_yticks(y,[NAMES[n] for n in model_names],fontsize=9)
        ax.invert_yaxis(); ax.set(xlim=(max(0,float(min(cv.min(),held.min())-.045)),1.005),xlabel="Macro-F1")
        ax.grid(axis="x",alpha=.17); ax.legend(fontsize=8,loc="lower center",bbox_to_anchor=(.4,-.45),ncol=2,frameon=False)
        for i,n in enumerate(model_names):
            ax.text(1.002,i,f"{cv[i]:.4f}",fontsize=8,ha="right",va="center",color=COLORS[0])
        t(fig,.065,.525,f"Обрано: {NAMES[selected]}. CV = {number(final['cv_mean'])} ± {number(final['cv_std'])}.\n"
          f"Контроль = {number(final['holdout_macro_f1'])}; вага енкодера = {number(m['transformer_weight'],2)}.",10,"bold")
        direction1="підтримана" if delta_char>0 else "не підтверджена"
        direction2="підтримана" if delta_chunk>0 else "не підтверджена"
        t(fig,.065,.470,
          f"Гіпотеза 1 {direction1}: зміна CV відносно слів — {delta_char:+.4f}.\n"
          f"Гіпотеза 2 {direction2}: зміна CV відносно початку — {delta_chunk:+.4f}.\n"
          "Висновки стосуються цих налаштувань і бюджету навчання, а не всіх моделей.",9.5)
        heading(fig,.397,5,"Помилки підсумкової моделі")
        ax=fig.add_axes([.15,.170,.34,.185])
        cm=np.array(final["holdout_confusion"],float); rates=cm/cm.sum(1,keepdims=True)
        ax.imshow(rates,cmap="Blues",vmin=0,vmax=1,aspect="auto")
        ax.set_xticks(range(3),LABELS); ax.set_yticks(range(3),LABELS)
        ax.set(xlabel="Передбачений клас",ylabel="Справжній клас")
        for i in range(3):
            for j in range(3):
                ax.text(j,i,number(rates[i,j]*100,1)+"%",ha="center",va="center",fontsize=9,
                        color="white" if rates[i,j]>.55 else INK)
        report=final["holdout_report"]
        t(fig,.56,.355,f"F1 human: {number(report['human']['f1-score'])}\n"
          f"F1 ai: {number(report['ai']['f1-score'])}\nF1 fusion: {number(report['fusion']['f1-score'])}\n\n"
          f"Fusion → human: {number(rates[2,0]*100,1)}%.\n"
          "Рядок матриці = 100%.\nMacro-F1 однаково враховує всі класи;\nце не відсоток правильних відповідей.",9.5)
        t(fig,.065,.111,"Межі: наближене групування, одна епоха енкодера та до трьох частин тексту.\n"
          "CV використано для вибору моделі, тому це не незалежна вкладена оцінка.\n"
          "Контроль спільний для порівнянь; остаточна перевірка узагальнення — private LB.",9,color=MUTED)
        footer(fig,2); pdf.savefig(fig); plt.close(fig)

        fig=plt.figure(figsize=(8.27,11.69),facecolor="white")
        t(fig,.065,.955,"Візуальний додаток",20,"bold")
        t(fig,.065,.917,"Пояснення рішень, довгі тексти та відтворення",11,color=MUTED)
        heading(fig,.865,6,"Що саме впливає на передбачення")
        ex=attribution["examples"][0]
        features=sorted(ex["top_features"],key=lambda x:abs(x["contribution"]))[-7:]
        ax=fig.add_axes([.19,.622,.70,.18])
        values=[x["contribution"] for x in features]
        ax.barh([repr(x["feature"]) for x in features],values,color=[COLORS[0] if x>=0 else COLORS[1] for x in values])
        ax.axvline(0,color=MUTED,lw=.8)
        ax.set_xlabel(f"Внесок у різницю оцінок: {ex['prediction']} мінус {ex['runner_up']}")
        ax.set_title("Точні лінійні внески символьної моделі: один приклад",fontsize=10)
        t(fig,.065,.570,"Додатний внесок підтримує передбачений клас; від’ємний — альтернативний.\n"
          "Сума всіх внесків і зсуву точно відтворює різницю оцінок моделі.\n"
          "Це пояснює символьний компонент, а не доводить причинні ознаки AI-авторства.\n"
          "Символьні ознаки можуть також відображати тему, мову та джерело.",9.5)
        ax=fig.add_axes([.15,.330,.34,.15])
        cov=[token["mean_head_coverage"],token["mean_three_chunk_coverage"]]
        bars=ax.bar(["Початок","3 частини"],np.array(cov)*100,color=[COLORS[0],COLORS[2]],width=.55)
        ax.set(ylim=(0,110),ylabel="Середнє покриття токенів, %")
        ax.bar_label(bars,labels=[number(x*100,1)+"%" for x in cov],padding=3,fontsize=9)
        ax.set_title("Скільки тексту бачить енкодер",fontsize=10)
        t(fig,.56,.475,f"Медіана: {number(token['median_full_tokens'],0)} токенів.\n"
          f"Довші за один фрагмент:\n{number(token['fraction_exceeding_one_chunk']*100,1)}% навчальних документів.\n\n"
          "Більше покриття не гарантує кращого F1.\n"
          "Перевірено ті самі ваги енкодера;\nокремо на частинах він не донавчався.",9.5)
        t(fig,.065,.268,"Відтворення",12,"bold")
        t(fig,.065,.239,"Код, фіксовані налаштування, ревізія ваг, результати й передбачення\n"
          "входять до архіву проєкту. README містить послідовність команд.\n"
          "Після підготовки даних у Colab: run_cloud.py → finalize_experiments.py.\n"
          f"Ваги енкодера: {config['model']}\nРевізія: {config['revision'][:12]}; середовище: PyTorch {config['torch']}.",9.4)
        t(fig,.065,.153,"Код і результати: ZIP-вкладення REPRODUCIBLE_PROJECT.zip у цьому PDF.",9,color=COLORS[0])
        if receipts.get("status")=="submitted":
            scores=[str(x.get("public_score","очікується")) for x in receipts.get("submissions",[])]
            status="Kaggle: подання виконано. Public score: "+", ".join(scores)+"."
        else:
            status="Файл для Kaggle перевірено: 25 000 унікальних ID, порядок і назви класів."
        t(fig,.065,.127,status,9)
        t(fig,.065,.087,"Джерела: Kaggle HW3",7.5,color=MUTED).set_url("https://www.kaggle.com/competitions/ai-generated-text-detection-hw3-ucu-ai-2026")
        t(fig,.34,.087,"scikit-learn",7.5,color=MUTED).set_url("https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction")
        t(fig,.51,.087,"DistilBERT",7.5,color=MUTED).set_url("https://huggingface.co/distilbert/distilbert-base-multilingual-cased")
        t(fig,.69,.087,"MinHash LSH",7.5,color=MUTED).set_url("https://ekzhu.com/datasketch/documentation.html")
        footer(fig,3); pdf.savefig(fig); plt.close(fig)
    # Standalone comparison figures for notebooks and the reproducibility archive.
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout="constrained")
    for i,name in enumerate(model_names):
        row=m["models"][name]
        axes[0].errorbar(row["cv_mean"],i,xerr=row["cv_std"],fmt="o",capsize=4,color=COLORS[0])
    axes[0].set_yticks(range(len(model_names)),[NAMES[n] for n in model_names]); axes[0].invert_yaxis()
    axes[0].set(title="Три групові розбиття: macro-F1 ± SD",xlabel="Macro-F1")
    for j,label in enumerate(LABELS):
        axes[1].bar(np.arange(len(model_names))+(j-1)*.25,
                    [m["models"][n]["holdout_report"][label]["f1-score"] for n in model_names],
                    width=.25,label=label,color=COLORS[j])
    axes[1].set_xticks(range(len(model_names)),[NAMES[n] for n in model_names],rotation=35,ha="right",fontsize=8)
    axes[1].set(title="Окремий контроль: F1 кожного класу",ylim=(0,1)); axes[1].legend()
    fig.savefig(ROOT/"results/final_comparison.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(5.5,4.5),layout="constrained")
    raw=np.array(final["holdout_confusion"],float)
    rates=raw/raw.sum(axis=1,keepdims=True)
    ax.imshow(rates,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(3),LABELS); ax.set_yticks(range(3),LABELS)
    ax.set(title="Підсумкова модель: контроль",xlabel="Передбачений клас",ylabel="Справжній клас")
    for i in range(3):
        for j in range(3):
            ax.text(j,i,f"{rates[i,j]*100:.1f}%",ha="center",va="center",
                    color="white" if rates[i,j]>.55 else INK)
    fig.savefig(ROOT/"results/final_confusion.png",dpi=160); plt.close(fig)
    report_text=["# Визначення походження тексту", "", "## Результати", "",
                 "| Модель | CV macro-F1 | SD | Контроль macro-F1 |", "|---|---:|---:|---:|"]
    for n in model_names:
        r=m["models"][n]; report_text.append(f"| {NAMES[n]} | {r['cv_mean']:.4f} | {r['cv_std']:.4f} | {r['holdout_macro_f1']:.4f} |")
    report_text += ["",f"Обрана модель: {NAMES[selected]}. Вибір за кросвалідацією; контроль не використано для вибору ваг.","",
         f"Символьні ознаки: зміна CV {delta_char:+.4f}. Читання трьох частин: зміна CV {delta_chunk:+.4f}.","",
         "Повний звіт із методикою, графіками, поясненнями та обмеженнями: FINAL_REPORT_UK.pdf."]
    (ROOT/"REPORT_UK.md").write_text("\n".join(report_text)+"\n")
    guide=f"""# Пояснення методики та результатів

## Дані й чесна перевірка

Модель обирає human, ai або fusion. Використано 40 000 навчальних текстів; усі системи отримали ті самі ID. Окремий контроль містить 54 469 документів.

Схожі версії документа можуть полегшувати перевірку, якщо їх розділити між навчанням і контролем. Тому знайдені лексично близькі тексти залишалися разом. Пошук наближений; сильно переписані версії могли залишитися невиявленими.

На трьох розбиттях модель щоразу навчалася на двох частинах і передбачала третю. Для кожного документа збережено передбачення моделі, яка його не бачила. Так отримано OOF-передбачення. Однакові розбиття зменшують вплив випадкового вибору даних на порівняння.

Macro-F1 надає однакову вагу всім трьом класам. Це не частка правильних відповідей. Стандартне відхилення трьох оцінок показує їхній розкид, а не довірчий інтервал.

## Слова та символи

TF-IDF перетворює документ на ваги ознак. Для слів використано окремі слова й пари слів, до 80 000 ознак. Для символів — фрагменти 3–5 знаків усередині меж слів, до 100 000 ознак, зі збереженням регістру. Символи можуть відображати написання, пунктуацію та частини слів, але також тему й мову.

Логістична регресія навчає ваги ознак для трьох класів. C=4 обрано після малого пошуку C=1/4. Більше C означає слабше обмеження ваг. Для порівняння представлень C залишався однаковим. Словник завжди навчався лише на навчальній частині конкретного розбиття.

Зміна CV при заміні слів на символи: {delta_char:+.4f}. Це результат цієї конфігурації; він не доводить, що символьний підхід завжди кращий або гірший.

## Трансформер і довгі тексти

Використано відкритий багатомовний DistilBERT. Він читає послідовність токенів і може враховувати контекст; TF-IDF працює з частотними ознаками всього документа. Тому порівняння стосується повних систем із різним покриттям тексту, а не тільки архітектур.

Токен не дорівнює слову. Послідовність містить до 254 змістових токенів і два службові. Зафіксовано одну епоху, learning rate 3e-5, batch 16 і накопичення градієнтів за два batch. Отже, один крок оптимізатора зазвичай враховує 32 документи. FP16 зменшує пам'ять і прискорює GPU-обчислення. Одна епоха — обмежений бюджет, а не доведений оптимум.

У варіанті chunking ті самі донавчені ваги застосовувалися до початку, середини й кінця документа. Усереднювалися ймовірності класів; це не голосування за клас більшістю фрагментів. Дуже довгі тексти все одно покриваються не повністю. Окремого донавчання на фрагментах не було.

Зміна CV від читання трьох частин замість початку: {delta_chunk:+.4f}. Середнє покриття токенів зросло з {token['mean_head_coverage']:.1%} до {token['mean_three_chunk_coverage']:.1%}.

## Ансамбль і вибір

Ансамбль усереднює ймовірності кількох моделей. Ваги 0, 0.25, 0.5, 0.75 і 1 порівняно за CV, без підбору на контрольних мітках. Для суміші слів і символів вага символів становить {char_weight:.2f}. Обраний класичний компонент: {m['classical_component']}; обраний енкодер: {m['transformer_component']}; його підсумкова вага: {m['transformer_weight']:.2f}.

Фінальна система: {selected}. CV macro-F1 = {final['cv_mean']:.4f} ± {final['cv_std']:.4f}; контрольний macro-F1 = {final['holdout_macro_f1']:.4f}. Обраний CV-результат частково оптимістичний, оскільки за ним робився вибір. Це не вкладена незалежна оцінка.

## Пояснюваність

Для символьного компонента обчислено точні доданки різниці оцінок передбаченого й другого класів: вага ознаки множиться на її TF-IDF. Додатні внески підтримують передбачений клас, від'ємні — альтернативний. Сума всіх внесків і різниці зсувів відтворює різницю оцінок; це перевірено чисельно.

Таке пояснення показує, на які ознаки спирається конкретна лінійна модель. Воно не пояснює весь ансамбль і не доводить причинну різницю між людськими та машинними текстами.

## Межі висновків

Навчання виконано на підвибірці, а не на всіх 272 352 документах. Частки класів, мови й джерела відрізняються. Лексичне групування може пропускати пов'язані версії. Енкодер навчався одну епоху й бачив обмежені фрагменти. Public leaderboard і локальний контроль не замінюють private leaderboard та перевірку на інших наборах.
"""
    (ROOT/"GUIDE_UK.md").write_text(guide)
    build_notebook(m)
    print("Built FINAL_REPORT_UK.pdf and final results notebook")


def build_notebook(metrics):
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    nb=nbf.v4.new_notebook()
    nb.metadata.kernelspec={"display_name":"Python 3","language":"python","name":"python3"}
    nb.cells=[
        md("# Визначення походження тексту\n\nКласи `human`, `ai`, `fusion`; англійська й німецька. "
           "Моделі порівнюються на тих самих 40 000 навчальних документах, трьох групових розбиттях "
           "і фіксованому контролі з 54 469 документів. Метрика — macro-F1."),
        md("## Відтворення\n\nREADME містить завантаження даних через Kaggle CLI, створення груп, "
           "підготовку фіксованої вибірки та запуск Colab. `classical_experiments.py` і "
           "`transformer_experiments.py` містять повний код навчання. `finalize_experiments.py` "
           "обирає модель за CV та перевіряє передбачення. Цей ноутбук читає фактичні збережені результати."),
        code("import json\nfrom pathlib import Path\nimport pandas as pd\nfrom IPython.display import Image, display\n"
             "ROOT = Path.cwd()\n"
             "def load(name):\n    return json.loads((ROOT / 'results' / name).read_text())\n"
             "metrics = load('final_metrics.json')\n"
             "audit = load('data_audit.json')"),
        md("## Дані та розбиття\n\nТочних повторів не виявлено. Лексично близькі тексти "
           "об'єднано через MinHash LSH і точний поріг схожості Жаккара 0,65. "
           "Початкові джерельні ID невідомі; сильні перефразування можуть бути пропущені."),
        code("display(Image(filename=str(ROOT / 'results' / 'eda.png')))\n"
             "display(Image(filename=str(ROOT / 'results' / 'train_test_shift.png')))\n"
             "display(pd.Series(load('split_audit.json')).drop(['limitations']).to_frame('значення'))"),
        md("### Приклади класів\n\nПерші уривки по одному для кожної комбінації мови й класу. "
           "Мітки взяті з train.csv. Уривок `fusion` може не містити переписаної частини; "
           "за коротким початком не можна встановлювати походження всього документа."),
        code("examples = {}\n"
             "for chunk in pd.read_csv(ROOT / 'data' / 'train.csv', chunksize=1000):\n"
             "    for row in chunk.itertuples():\n"
             "        key = (row.lang, row.label)\n"
             "        if key not in examples:\n"
             "            examples[key] = ' '.join(row.text.split())[:350]\n"
             "    if len(examples) == 6:\n        break\n"
             "for (lang, label), excerpt in sorted(examples.items()):\n"
             "    print(lang, label)\n"
             "    display(pd.DataFrame({'Уривок': [excerpt]}).style.format(escape='html').set_properties(**{'white-space':'normal'}))"),
        md("## Перевірені гіпотези\n\n1. Символьні фрагменти 3–5 знаків можуть допомагати розрізняти "
           "`human` і `fusion`. Їх порівняно зі словами за однакового C=4.\n"
           "2. Для того самого донавченого DistilBERT порівняно початок документа та середні ймовірності "
           "за початком, серединою й кінцем. Це перевірка додаткового контексту під час передбачення, "
           "без окремого донавчання на фрагментах.\n\n"
           "Енкодер: одна епоха, 256 токенів, LR=3e-5, batch=16, накопичення=2, FP16. "
           "Фактичний пристрій і версії наведені в конфігурації нижче."),
        code("display(pd.Series(load('transformer_config.json')).to_frame('налаштування'))\n"
             "display(pd.Series(load('development_tokenization.json')).to_frame('токенізація'))"),
        md("## Порівняння моделей\n\nCV використовується для вибору параметрів і ваг; контрольні "
           "оцінки наведені окремо. Стандартне відхилення трьох оцінок не є довірчим інтервалом. "
           "Вибраний CV-результат не є незалежною вкладеною оцінкою."),
        code("display(pd.DataFrame([{\n    'model': name, 'CV macro-F1': row['cv_mean'],\n"
             "    'CV SD': row['cv_std'], 'holdout macro-F1': row['holdout_macro_f1']\n"
             "} for name, row in metrics['models'].items()]).round(4))\n"
             "display(Image(filename=str(ROOT / 'results' / 'final_comparison.png')))\n"
             "display(pd.Series(load('hypotheses.json')).to_frame('результат'))"),
        md("## Помилки та підгрупи\n\nОкремі результати за мовами й джерелами допомагають "
           "помітити нерівномірну якість. Їхні розміри й баланси класів відрізняються."),
        code("selected = metrics['models'][metrics['selected_model']]\n"
             "display(pd.DataFrame({k: selected['holdout_report'][k] for k in ['human','ai','fusion','macro avg']}).T.round(4))\n"
             "display(pd.DataFrame(selected['holdout_confusion'], index=['human','ai','fusion'], columns=['human','ai','fusion']))\n"
             "display(Image(filename=str(ROOT / 'results' / 'final_confusion.png')))\n"
             "display(pd.DataFrame(selected['holdout_slices']).round(4))"),
        md("## Пояснюваність\n\nДля символьної моделі обчислено точний внесок TF-IDF-ознак "
           "у різницю оцінок передбаченого та другого класів. Перевірено, що сума внесків "
           "і зсуву відтворює цю різницю. Це локальне лінійне пояснення компонента; "
           "ознаки не доводять причинний механізм авторства."),
        code("display(Image(filename=str(ROOT / 'results' / 'character_attribution.png')))\n"
             "print('Обрана модель:', metrics['selected_model'])\n"
             "display(pd.DataFrame(metrics['submission_files']))"),
        md("## Обмеження та джерела\n\nНавчання на підвибірці, одна епоха енкодера, наближені "
           "джерельні групи та неповне покриття найдовших документів обмежують висновки. "
           "Public leaderboard не замінює незалежну перевірку.\n\n"
           "[Kaggle HW3](https://www.kaggle.com/competitions/ai-generated-text-detection-hw3-ucu-ai-2026), "
           "[scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction), "
           "[DistilBERT](https://huggingface.co/distilbert/distilbert-base-multilingual-cased), "
           "[datasketch](https://ekzhu.com/datasketch/documentation.html)."),
    ]
    nbf.validate(nb)
    nbf.write(nb,ROOT/"ai-generated-text-detection-hw3-ucu-ai-2026.ipynb")


if __name__=="__main__":
    main()
