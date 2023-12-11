"""
This file contains some functions used to analyze the data from requests and interventions.
"""

import re
import datetime as dt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from torch import Tensor
from transformers import AutoModel, AutoTokenizer
import torch.nn.functional as F


SUPPLIES_TAGS = {
        'alimentation': 'ALIMENTATION ET EAU / FOOD AND WATER / الغذاء والماء',
        'eau': 'ALIMENTATION ET EAU / FOOD AND WATER / الغذاء والماء',
        'food': 'ALIMENTATION ET EAU / FOOD AND WATER / الغذاء والماء',
        'water': 'ALIMENTATION ET EAU / FOOD AND WATER / الغذاء والماء',
        'nourriture': 'ALIMENTATION ET EAU / FOOD AND WATER / الغذاء والماء',
        'medical': 'ASSISTANCE MÉDICALE / MEDICAL ASSISTANCE / المساعدة الطبية',
        'médical': 'ASSISTANCE MÉDICALE / MEDICAL ASSISTANCE / المساعدة الطبية',
        'doctor': 'ASSISTANCE MÉDICALE / MEDICAL ASSISTANCE / المساعدة الطبية',
        'vêtements': 'VÊTEMENTS / CLOTHES / الملابس',
        'clothes': 'VÊTEMENTS / CLOTHES / الملابس',
        'secours': 'SECOURS / RESCUE / الإنقاذ',
        'rescue': 'SECOURS / RESCUE / الإنقاذ',
        'refuge': 'REFUGE / SHELTER / المأوى',
        'shelter': 'REFUGE / SHELTER / المأوى',
        'couvertures': 'COUVERTURES / COVERS / البطانيات',
        'covers': 'COUVERTURES / COVERS / البطانيات',
        'pharmaceuticals': 'PHARMACEUTICALS / MEDICAMENTS / الأدوية',
        'medicaments': 'PHARMACEUTICALS / MEDICAMENTS / الأدوية',
        'pharmacy': 'PHARMACEUTICALS / MEDICAMENTS / الأدوية',
        'medicine': 'PHARMACEUTICALS / MEDICAMENTS / الأدوية',
        'blankets': 'COUVERTURES / COVERS / البطانيات',
        'tents': 'REFUGE / SHELTER / المأوى',
        'couches': 'PHARMACEUTICALS / MEDICAMENTS / الأدوية'
    }

SUPPLIES_NEEDS_CATEGORIES = ['ALIMENTATION ET EAU / FOOD AND WATER / الغذاء والماء',
                       'ASSISTANCE MÉDICALE / MEDICAL ASSISTANCE / المساعدة الطبية',
                       'VÊTEMENTS / CLOTHES / الملابس',
                       'SECOURS / RESCUE / الإنقاذ',
                       'REFUGE / SHELTER / المأوى',
                       'COUVERTURES / COVERS / البطانيات',
                       # 'KITCHEN TOOLS / USTENSILES DE CUISINE / أدوات المطبخ',
                       'PHARMACEUTICALS / MEDICAMENTS / الأدوية',
                       'OTHER']

TRANSLATION_DICT = {
    'أغطية': 'covers',
    'أسرة': 'beds',
    'وسادات': 'pillows',
    'مصابح': 'lamps',
    'خيام': 'tents',
    'ألعاب أطفال': 'toys',
    'قليل من المواد الغذائية': 'food',
    'افرشة': 'covers',
    'جلباب': 'clothes',
    'ملابس': 'clothes',
    'لديهم كل شيء': 'unknown'
}


def clean_text(text):
    """
    remove special characters from text
    """
    pattern = re.compile(r'[\u200e\xa0()\u200f]')
    cleaned_text = pattern.sub('', text)
    return cleaned_text


def contains_arabic(text):
    """
    check if the text contains arabic characters
    """
    arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
    if type(text)!=str:
      return False
    return arabic_pattern.search(text) is not None


def arabic_to_latin_punctuation(text):
    """
    replace arabic punctuation with latin punctuation
    """
    punctuation_mapping = {
        '،': ',',
        '؛': ';',
        'ـ': '_',
        '؟': '?',
        '٪': '%',
        '٫': '.',
    }

    for arabic_punct, latin_punct in punctuation_mapping.items():
        text = text.replace(arabic_punct, latin_punct)

    return text


def plot_timeline(df: pd.DataFrame, today: dt.datetime, date_col: str):
    """Plot the timeline of requests and interventions.
    """
    df_past = df[df[date_col]<=today.date()]
    df_future = df[df[date_col]>today.date()]

    count_past = (df_past
                  .groupby(date_col)
                  .size()
                  .rename('count')
                  .reset_index())
    past_date_range = pd.date_range(start=min(count_past[date_col]), 
                                    end=today.date(), 
                                    freq='D')
    count_past = (count_past
                  .set_index(date_col)
                  .reindex(past_date_range, fill_value=0)
                  .reset_index())

    if len(df_future)>0:
        count_future = df_future.groupby(date_col).size().rename('count').reset_index()
        future_date_range = pd.date_range(start=today.date()+dt.timedelta(days=1), 
                                          end=max(count_future[date_col]), 
                                          freq='D')
        count_future = (count_future
                        .set_index(date_col)
                        .reindex(future_date_range, fill_value=0)
                        .reset_index())
    else:
        count_future = pd.DataFrame()

    bridge_date = today.date()
    bridge_data = pd.DataFrame(
        {'index': bridge_date, 'form_date':count_past.iloc[-1]['count']}, index=[0])
    count_future = pd.concat([bridge_data, count_future], ignore_index=True)

    # Plot
    fig = go.Figure()
    # past 
    fig.add_trace(go.Scatter(x=count_past['index'], 
                             y=count_past['count'], 
                             mode='lines',
                             name='Past Interventions', 
                             line=dict(color='blue')))
    # future
    fig.add_trace(go.Scatter(x=count_future['index'], 
                             y=count_future['count'], 
                             mode='lines',
                             name='Future Interventions', 
                             line=dict(color='orange')))

    fig.add_vline(x=today.date(), line_dash="dash", line_color="black")

    fig.update_layout(yaxis_title="#", xaxis_title='date')
    return fig


def classify_supplies_rule_based(text: pd.DataFrame, keep_raw: bool = False):
    """ Classifies text into supplies categories from SUPPLIES_TAGS
      using a rule-based approach."""
    classes = []
    lowercase_text = text.lower()  # case-insensitive matching

    for keyword, category in SUPPLIES_TAGS.items():
        if keyword in lowercase_text:
            classes.append(category)

    if keep_raw:
        classes.append(lowercase_text)

    elif not classes:
        classes.append('OTHER')

    return list(set(classes))


def classify_multilingual_field_e5(df: pd.DataFrame,
                      field_to_tag: str = 'supplies', 
                      categories: list = SUPPLIES_NEEDS_CATEGORIES):
    """
    Tag supplies/requests into categories using multilingual-e5-large model.
    Returns a dataframe with a new column containing the list of predicted categories.
    Requires CUDA
    """
    def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
    model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')
    model.cuda()

    # classify ar supplies
    processed_df = df.copy()
    values_to_classify = processed_df[field_to_tag]

    mapped_inputs = dict()

    for text in values_to_classify:
        gt = [f"{s}" for s in categories]
        qr = [f"{v}" for v in re.split("\.|,| و", text)]
        input_texts = qr + gt

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.cuda() for k, v in batch_dict.items()}

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[:len(qr)] @ embeddings[len(qr):].T) * 100

    mapped_inputs[text] = list(
        set([categories[int(scores[i,:].argmax())] for i in range(len(qr))]))

    processed_df.loc[values_to_classify.index, f'{field_to_tag}_category'] = list(
        mapped_inputs.values())
    
    return processed_df


def plot_categories_share(raw_df: pd.DataFrame, 
                          today: dt.datetime, 
                          field: str = 'supplies'):
    """
    Plot the share of each category of requests/supplies.
    """
    df = raw_df[[field, f'{field}_category']].explode(f'{field}_category')
    pie_data = df.groupby(f'{field}_category', as_index=False).size().rename('n')
    fig = px.pie(pie_data, 
                 names=f'{field}_category', 
                 values='n', 
                 title=f'# per {field} category up till {today.date()}',
                labels={f'{field}_category': f'{field}', 'n': '%'})
    return fig