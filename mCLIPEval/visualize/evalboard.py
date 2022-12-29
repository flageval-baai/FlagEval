from .constants import _DATASET_INFO
import streamlit as st
import plotly.express as px
import json
import pandas as pd
from PIL import Image
import warnings



class Radar(object):
    def __init__(self, score_list, width=None, height=None, theta_col='dataset') -> None:
        self.norm_score_list = self.normalize(score_list, group_by=theta_col)
        self.width = width
        self.height = height
        self.theta_col = theta_col
        warnings.filterwarnings("ignore")
    def normalize(self, score_list, group_by):
        assert isinstance(score_list, list) and len(score_list)>0
        assert isinstance(score_list[0], dict) and group_by in score_list[0].keys()
        groups = {}
        for _element in score_list:
            groups[_element[group_by]] = groups.get(_element[group_by], [])
            groups[_element[group_by]].append(_element['score'])
        def calc_norm_score(score, max_series, min_series):
            if isinstance(score, list) and isinstance(max_series, list) and isinstance(min_series, list):
                assert len(score) == len(max_series) and len(score) == len(min_series)
                return [calc_norm_score(s, ma, mi) for s, ma, mi in zip(score, max_series, min_series)]
            if score == max_series and score == min_series:
                return 50
            return 80*(score-min_series)/(max_series-min_series) + 10
        norm_score_list = []
        for _element in score_list:
            series = groups[_element[group_by]]
            _element['norm'] = calc_norm_score(_element['score'], max(series), min(series))
            norm_score_list.append(_element)
        return norm_score_list
    def get_plotly_figure(self):
        df = pd.DataFrame(self.norm_score_list)
        fig = px.line_polar(
            df, 
            r='norm', 
            theta=self.theta_col, 
            color='model', 
            hover_name='model', 
            hover_data={'model':False, 'norm': False, self.theta_col:True,'score':True}, 
            line_close=True, 
            markers=True,
            range_r=[0, 100],
            start_angle=0,
            width=self.width,
            height=self.height
        )
        fig.update_layout(polar = dict(radialaxis = dict(visible = False)))
        # fig.update_layout(polar = dict(radialaxis = dict(showticklabels = False)))

        return fig

class EvaluationBoard(object):
    def __init__(self, eval_jsonl, multi_input=False) -> None:
        st.set_page_config(layout='wide')
        image_bg = Image.open("../logo.png")
        col_left, col_mid, col_right = st.columns(3)
        with col_mid:
            st.image(image_bg, width=512, use_column_width=True)
        st.markdown("**mCLIPEval** is an multilingual and multi-task evaluation tookit for CLIP (Contrastive Language–Image Pre-training) model series. The project is supported by FlagEval Team from BAAI. The associated resources are open-sourced in [FlagEval project](https://github.com/FlagOpen/FlagEval).")
        self.initialize_with_jsonl_file(eval_jsonl, multi_input)
    
    def initialize_with_jsonl_file(self, eval_jsonl, multi_input):
        eval_results = [] 
        self.datasets = set()
        if not multi_input:
            with open(eval_jsonl) as fd:
                for line in fd:
                    result = json.loads(line.strip().replace('NaN', 'null'))
                    eval_results.append(result)
                    datasets = [key for key in result.keys() if key!='model_info']
                    self.datasets.update(datasets)
        else:
            ori_eval_json_list = eval_jsonl.split(',')
            eval_json_list = []
            import glob
            for ori_eval_json in ori_eval_json_list:
                eval_json_list += glob.glob(ori_eval_json)
            print('Visualization of the following results files: '+ ', '.join(eval_json_list))
            for eval_json in eval_json_list:
                with open(eval_json) as fd:
                    result = json.loads(fd.read().strip().replace('NaN', 'null'))
                    eval_results.append(result)
                    datasets = [key for key in result.keys() if key!='model_info']
                    self.datasets.update(datasets)
        self.initialize_settings(eval_results)
        self.parse_raw_results(eval_results)

    def initialize_settings(self, eval_results):
        def get_dataset_attr(key):
            full_attr_list = [_DATASET_INFO.get(dataset, None) for dataset in self.datasets]
            attr_list = [full_attr.get(key, None) for full_attr in full_attr_list if isinstance(full_attr, dict)]
            attr_list = list(set([attr for attr in attr_list if attr]))
            return attr_list
        self.languages = get_dataset_attr('language')
        self.languages_default = ["CN","EN"]
        self.groups = get_dataset_attr('group')
        self.modals = get_dataset_attr('modal')
        self.tasks = get_dataset_attr('task')
        if 'Image-Text Retrieval' in self.tasks:
            self.tasks.remove('Image-Text Retrieval')
            self.tasks.append('Image Retrieval')
            self.tasks.append('Text Retrieval')
        def get_model_attr(key):
            full_attr_list = [eval_result.get('model_info') for eval_result in eval_results]
            attr_list = [full_attr.get(key, None) for full_attr in full_attr_list if isinstance(full_attr, dict)]
            attr_list = list(set([attr for attr in attr_list if attr]))
            return attr_list
        self.agencies = get_model_attr('agency')
        self.vision_encoders = get_model_attr('vision_encoder')
        self.text_encoders = get_model_attr('text_encoder')
    
    def parse_raw_results(self, eval_results):
        def calc_score(metrics, _group):
            def norm_score(score):
                if isinstance(score, list):
                    return [norm_score(s) for s in score]
                return round(100.0 * score, 2)
            score = 0.0
            if len(metrics)==1:
                for _, value in metrics.items():
                    score = value
                    break
                return norm_score(score)
            elif _group == "RETRIEVAL":
                return norm_score([metrics.get('IR@1', 0.0), metrics.get('TR@1', 0.0)])
            elif _group == "CLASSIFICATION":
                return norm_score(metrics.get('acc1', 0.0))
            else:
                return norm_score(metrics.get('group_score', 0.0))
        data_dict_list = []
        for eval_res in eval_results:
            model_info = eval_res.get('model_info', {})
            model_name = model_info.get('model_name', None)
            vision_encoder = model_info.get('vision_encoder', 'unknown')
            text_encoder = model_info.get('text_encoder', 'unknown')
            agency = model_info.get('agency', 'unknown')
            for key, value in eval_res.items():
                if key!='model_info':
                    dataset_name = key
                    dataset_info = _DATASET_INFO.get(dataset_name, {})
                    language = dataset_info['language']
                    group = dataset_info['group']
                    task = dataset_info['task']
                    score = calc_score(value, group)
                    if group == "RETRIEVAL":
                        data_dict_list.append(
                            {
                                'model': model_name,
                                'dataset': dataset_name + ' T2I',
                                'language': language,
                                'group': group,
                                'task': 'Image Retrieval',
                                'metrics': value,
                                'score': score[0],
                                'vision_encoder': vision_encoder,
                                'text_encoder': text_encoder,
                                'agency': agency
                            }
                        )
                        data_dict_list.append(
                            {
                                'model': model_name,
                                'dataset': dataset_name + ' I2T',
                                'language': language,
                                'group': group,
                                'task': 'Text Retrieval',
                                'metrics': value,
                                'score': score[1],
                                'vision_encoder': vision_encoder,
                                'text_encoder': text_encoder,
                                'agency': agency
                            }
                        )
                    else:
                        data_dict_list.append(
                            {
                                'model': model_name,
                                'dataset': dataset_name,
                                'language': language,
                                'group': group,
                                'task': task,
                                'metrics': value,
                                'score': score,
                                'vision_encoder': vision_encoder,
                                'text_encoder': text_encoder,
                                'agency': agency
                            }
                        )
        self.data_dict_list = sorted(data_dict_list, key=lambda x:(x['model'], x['dataset']))

    def get_conditional_data(self, conditions=None):
        filtered_list = []
        def check_conditions(dic, conditions):
            for key, value in conditions.items():
                if not value or len(value) == 0:
                    continue
                if dic[key] in value:
                    continue
                if key=='dataset':
                    _value = dic[key].replace(' I2T', '').replace(' T2I', '')
                    if _value in value:
                        continue
                return False
            return True
        for data_dict in self.data_dict_list:
            if check_conditions(data_dict, conditions):
                filtered_list.append(data_dict)
        return filtered_list

    def _to_data_dict_group(self, data_dict_list, group_by='dataset'):
        group_data_dict = []
        num_samples = {}
        sum_scores = {}
        for data_dict in data_dict_list:
            key = (data_dict['model'], data_dict[group_by])
            sum_scores[key] = sum_scores.get(key, 0) + data_dict['score']
            num_samples[key[1]] = num_samples.get(key[1], {})
            num_samples[key[1]][key[0]] = num_samples[key[1]].get(key[0], 0) + 1
        for key, score in sum_scores.items():
            model_name, group = key
            score = score*1.0/max(num_samples[group].values())
            group_data_dict.append(
                {
                    'model': model_name,
                    group_by: group,
                    'score': score
                }
            )
        return group_data_dict

    def _to_leaderboard_dataframe(self, data_dict_list):
        df_dict = {}
        for data_dict in data_dict_list:
            dataset = data_dict['dataset']
            if data_dict['agency']=='unknown' or not data_dict['agency']:
                model = data_dict['model']
            else:
                model = data_dict['model']+ ' '+ data_dict['agency']
            df_dict[dataset] = df_dict.get(dataset, {})
            df_dict[dataset][model] = data_dict['score']
        
        df = pd.DataFrame(df_dict)
        df['mean'] = df.mean(axis=1)
        cols = list(df.columns.values)
        cols = cols[-1:]+cols[:-1]
        return df[cols].sort_values(by='mean', ascending=False)
            
    def _to_comparison_dataframe(self, selected_datasets, model1, model2):
        data_dict_list = self.get_conditional_data(conditions={'dataset': selected_datasets, 'model':[model1, model2]})
        df_dict = {}
        for data_dict in data_dict_list:
            df_dict[data_dict['dataset']] = df_dict.get(data_dict['dataset'], 0)
            if model1 != model2:
                if data_dict['model'] == model1:
                    df_dict[data_dict['dataset']] += data_dict['score']
                else:
                    df_dict[data_dict['dataset']] -= data_dict['score']
            df_dict[data_dict['dataset']] = round(df_dict[data_dict['dataset']], 2)
        
        df = pd.DataFrame(df_dict, index=[f'Δ Score (%)'])
        df = pd.DataFrame(df.values.T, index=df_dict.keys(), columns=df.index).sort_values(by=f'Δ Score (%)')
        return df

    def _to_single_dataframe(self, selected_dataset):
        data_dict_list = self.get_conditional_data(conditions={'dataset': [selected_dataset]})
        df_dict = {}
        for data_dict in data_dict_list:
            metrics = data_dict['metrics']
            model = data_dict['model']
            for key, value in metrics.items():
                metric = key
                score = round(100.0 * value, 2)
                df_dict[metric] = df_dict.get(metric, {})
                df_dict[metric][model] = score
        df = pd.DataFrame(df_dict)
        return df

    
    def visualize(self):
        st.header('mCLIPEval Evaluation Board')
        table_options = ['Leaderboard', 'Model vs. Model', 'Single Dataset']
        tab1, tab2, tab3 = st.tabs(table_options)
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                selected_langs = st.multiselect('Dataset Languages: ', sorted(self.languages), sorted(self.languages))
                selected_groups = st.multiselect('Dataset Groups:', sorted(self.groups), sorted(self.groups))
                selected_tasks = st.multiselect('Dataset Tasks:', sorted(self.tasks), sorted(self.tasks))
            with col2:
                selected_agencies = st.multiselect('Model Agencies:', sorted(self.agencies), sorted(self.agencies))
                selected_text_encoders = st.multiselect('Model Text Encoders:', sorted(self.text_encoders), sorted(self.text_encoders))
                selected_vision_encoders = st.multiselect('Model Vision Encoders:', sorted(self.vision_encoders), sorted(self.vision_encoders))
            
            conditions = {
                'language': selected_langs,
                'group': selected_groups,
                'task': selected_tasks,
                'agency': selected_agencies,
                'text_encoder': selected_text_encoders,
                'vision_encoder': selected_vision_encoders
            }
            data_dict_1 = self.get_conditional_data(conditions)
            full_models = list(set([ele['model'] for ele in data_dict_1]))
            full_datasets = list(set([ele['dataset'] for ele in data_dict_1]))
            with st.expander('CLICK TO EXPAND TO SPECIFY MODELS AND DATASETS:'):
                col_left, col_right = st.columns(2)
                with col_left:
                    selected_models = st.multiselect('Choose Models: ', sorted(full_models), sorted(full_models))
                with col_right:
                    selected_datasets = st.multiselect('Choose Datasets for Comparisons: ', sorted(full_datasets), sorted(full_datasets))
            conditions['model'] = selected_models
            conditions['dataset'] = selected_datasets
            data_dict_2 = self.get_conditional_data(conditions)
            selected_chart = st.select_slider(
                "Slide to Select Radar Chart You Need (Options: Languages, Task Groups, Tasks, Datasets)",
                options=['Language Radar Chart', 'Group Radar Chart', 'Task Radar Chart','Dataset Radar Chart'],
                value='Language Radar Chart'
            )
            group_by = selected_chart[:selected_chart.index(' ')].lower()
            data_dict_3 = self._to_data_dict_group(data_dict_2, group_by=group_by)
            ra1 = Radar(score_list=data_dict_3, width=800, height=800, theta_col=group_by)
            st.plotly_chart(ra1.get_plotly_figure(), use_container_width=True)
            df = self._to_leaderboard_dataframe(data_dict_2)
            fig = px.scatter(data_dict_2, x='dataset', y='score', height=1000, symbol='model', color='model')
            fig.update_xaxes(title='Dataset')
            fig.update_yaxes(title='Score (%)')
            fig.update_layout(legend_title='Model')
            st.plotly_chart(fig, use_container_width=True)
            # st.line_chart(df_t)
            st.dataframe(df)
            
        with tab2:
            col_lside, col1, col2, _ = st.columns(4)
            
            full_models = sorted(list(set([ele['model'] for ele in self.data_dict_list])))
            with col1:
                model1 = st.selectbox('Select 1st Model :', full_models)
            with col2:
                idx = len(full_models)-1 if len(full_models)>1 else 0
                model2 = st.selectbox('Select 2nd Model:', full_models, index=idx)
            with col_lside:
                st.markdown("")
                st.markdown("#### **Comparisons dataset by dataset**")
            full_datasets = list(set([ele['dataset'] for ele in self.data_dict_list]))
            with st.expander('CLICK TO EXPAND TO SPECIFY DATASETS:'):
                selected_datasets2 = st.multiselect('Choose Datasets: ', sorted(full_datasets), sorted(full_datasets))
            compare_df = self._to_comparison_dataframe(selected_datasets2, model1, model2)
            fig = px.bar(
                compare_df, 
                orientation='h',
                width=1500,
                height=1000,
                x = f'Δ Score (%)'
            )
            colors = ['green' if val[0]>0 else 'navy' for val in compare_df.values]
            fig.update_traces(marker_color=colors)
            fig.update_xaxes(title=f'Δ Score (%) <br> {model1} vs. {model2}')
            fig.update_yaxes(title='Dataset')
            st.plotly_chart(fig)
        
        with tab3:
            _, col_mid, _ = st.columns([1,3,1])
            full_datasets = list(set([ele['dataset'] for ele in self.data_dict_list]))
            with col_mid:
                dataset_name = st.selectbox("Specify the Detailed Dataset:", options=sorted(full_datasets))
                single_df = self._to_single_dataframe(selected_dataset=dataset_name)
                fig = px.bar(single_df, barmode='group', width=1000, height=800)
                fig.update_xaxes(title=f'Model')
                fig.update_yaxes(title='Score (%)')
                fig.update_layout(legend_title='Metric')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(single_df, use_container_width=True)

if __name__ == "__main__":
    eb = EvaluationBoard(eval_jsonl='outputs/*.json', multi_input=True)
    eb.visualize()
