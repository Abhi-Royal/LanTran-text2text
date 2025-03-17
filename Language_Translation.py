from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM

model_id = 'facebook/mbart-large-50-many-to-many-mmt'
key = "your huggingface key"
tokenizer =  AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipeline = pipeline("translation",model = model, tokenizer=tokenizer)

languages = {'Arabic':'ar_AR', 'Czech':'cs_CZ', 'German':'de_DE', 'English':'en_XX', 'Spanish':'es_XX','Estonian':'et_EE', 
             'Finnish': 'fi_FI', 'French' :'fr_XX', 'Gujarati' :'gu_IN', 'Hindi':'hi_IN', 'Italian':'it_IT', 'Japanese':'ja_XX', 
             'Kazakh' :'kk_KZ', 'Korean':'ko_KR', 'Lithuanian':'lt_LT', 'Latvian':'lv_LV', 'Burmese':'my_MM', 'Nepali':'ne_NP', 
             'Dutch':'nl_XX', 'Romanian':'ro_RO', 'Russian':'ru_RU', 'Sinhala':'si_LK', 'Turkish':'tr_TR','Vietnamese':'vi_VN', 
             'Chinese':'zh_CN','Afrikaans':'af_ZA','Azerbaijani':'az_AZ', 'Bengali':'bn_IN', 'Persian':'fa_IR', 'Hebrew':'he_IL',
             'Croatian':'hr_HR', 'Indonesian':'id_ID', 'Georgian':'ka_GE', 'Khmer':'km_KH','Macedonian':'mk_MK', 'Malayalam':'ml_IN', 
             'Mongolian':'mn_MN', 'Marathi':'mr_IN','Polish':'pl_PL', 'Pashto':'ps_AF', 'Portuguese':'pt_XX', 'Swedish':'sv_SE', 
             'Swahili':'sw_KE', 'Tamil':'ta_IN', 'Telugu':'te_IN', 'Thai':'th_TH', 'Tagalog':'tl_XX', 'Ukrainian':'uk_UA', 
             'Urdu':'ur_PK', 'Xhosa':' xh_ZA', 'Galician':'gl_ES', 'Slovene':'sl_SI'}

list_lang = list(languages.keys())
"""list_lang = ['Arabic', 'Czech', 'German', 'English', 'Spanish', 'Estonian', 'Finnish', 'French', 'Gujarati', 'Hindi', 'Italian', 'Japanese', 
             'Kazakh', 'Korean', 'Lithuanian', 'Latvian', 'Burmese', 'Nepali', 'Dutch', 'Romanian', 'Russian', 'Sinhala', 'Turkish', 
             'Vietnamese', 'Chinese', 'Afrikaans', 'Azerbaijani', 'Bengali', 'Persian', 'Hebrew', 'Croatian', 'Indonesian', 'Georgian', 
             'Khmer', 'Macedonian', 'Malayalam', 'Mongolian', 'Marathi', 'Polish', 'Pashto', 'Portuguese', 'Swedish', 'Swahili', 'Tamil', 
             'Telugu', 'Thai', 'Tagalog', 'Ukrainian', 'Urdu', 'Xhosa', 'Galician', 'Slovene']"""

list_lan_codes = list(languages.values())
"""list_lan_codes = ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT', 'ja_XX', 'kk_KZ', 'ko_KR', 
                  'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK', 'tr_TR', 'vi_VN', 'zh_CN', 'af_ZA', 'az_AZ', 'bn_IN', 'fa_IR',
                  'he_IL', 'hr_HR', 'id_ID', 'ka_GE', 'km_KH', 'mk_MK', 'ml_IN', 'mn_MN', 'mr_IN', 'pl_PL', 'ps_AF', 'pt_XX', 'sv_SE', 'sw_KE', 'ta_IN',
                  'te_IN', 'th_TH', 'tl_XX', 'uk_UA', 'ur_PK', 'xh_ZA', 'gl_ES', 'sl_SI']"""

while True:
    src_lang = input(f"Translate from:").capitalize()
    if src_lang in languages:
        source = list_lan_codes[list_lang.index(src_lang)]
    else:
        print("Language is not available")

    tgt_lang = input(f"Translate to: ").capitalize()
    if tgt_lang in languages:
        target = list_lan_codes[list_lang.index(tgt_lang)]
    else:
        print("Language is not in the list")

    user_input = input(f"Sentence to Translate: ")

    result = pipeline(user_input, src_lang=source, tgt_lang=target)

    # Print the translated sentence
    print("Translated Sentence:", result[0]['translation_text'])





