from .constants import _DATASET_INFO
import streamlit as st
import altair as alt
import json
import pandas as pd

class EvaluationBoard(object):
    def __init__(self, eval_jsonl, multi_input=False):
        st.set_page_config(layout="wide")
        self.initialize_with_jsonl_file(eval_jsonl, multi_input)
        self.initialize_settings()

    def initialize_with_jsonl_file(self, eval_jsonl, multi_input):
        self.eval_results = []
        self.datasets = set()
        if not multi_input:
            with open(eval_jsonl) as fd:
                for line in fd:
                    result = json.loads(line.strip().replace('NaN', 'null'))
                    self.eval_results.append(result)
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
                    result = json.loads(fd.read().strip())
                    self.eval_results.append(result)
                    datasets = [key for key in result.keys() if key!='model_info']
                    self.datasets.update(datasets)

    def initialize_settings(self):
        def get_dataset_attr(key):
            full_attr_list = [_DATASET_INFO.get(dataset, None) for dataset in self.datasets]
            attr_list = [full_attr.get(key, None) for full_attr in full_attr_list if isinstance(full_attr, dict)]
            attr_list = list(set([attr for attr in attr_list if attr]))
            return attr_list
        self.languages = get_dataset_attr('language')
        self.groups = get_dataset_attr('group')
        self.modals = get_dataset_attr('modal')
        self.tasks = get_dataset_attr('task')
        def get_model_attr(key):
            full_attr_list = [eval_result.get('model_info') for eval_result in self.eval_results]
            attr_list = [full_attr.get(key, None) for full_attr in full_attr_list if isinstance(full_attr, dict)]
            attr_list = list(set([attr for attr in attr_list if attr]))
            return attr_list
        self.agencies = get_model_attr('agency')
        self.vision_encoders = get_model_attr('vision_encoder')
        self.text_encoders = get_model_attr('text_encoder')

    def _to_dataframe(self, model_conditions=None, dataset_conditions=None, selected_datasets=None, selected_models=None):
        eval_results = self.eval_results
        if model_conditions:
            for key, value in model_conditions:
                eval_results = [res for res in eval_results if res['model_info'].get(key) in value]
        if selected_models:
            eval_results = [
                res for res in eval_results
                if res['model_info'].get('model_name')+'-'+ res['model_info'].get('agency')
                   in selected_models
            ]

        if dataset_conditions:
            for key, value in dataset_conditions:
                eval_results = [{
                    dataset: metrics for dataset, metrics in res.items()
                    if dataset == 'model_info'
                       or _DATASET_INFO[dataset][key] in value
                    } for res in eval_results]
        if selected_datasets:
            eval_results = [{
                dataset: metrics for dataset, metrics in res.items()
                if dataset == 'model_info'
                       or dataset in selected_datasets
                } for res in eval_results]
        def main_metric_socre(metric_dict):
            keys = set(metric_dict.keys())
            if 'acc1' in keys:
                score = metric_dict.get('acc1')*100
                return score, '%.2f'%score
            elif 'TR@1' in keys and 'IR@1' in keys:
                score1 = metric_dict.get('TR@1')*100
                score2 = metric_dict.get('IR@1')*100
                return (score1, score2), '%.2f'%score1 +'/'+'%.2f'%score2
            elif 'group_score' in keys:
                score = metric_dict.get('group_score')*100
                return score, '%.2f'%score
            else:
                score = metric_dict.get('mean_average_precision') * 100
                return score, '%.2f'%score
        indexes = []
        tab_dict_vals = []
        fig_dict_vals = []
        tab_col_set = set()
        fig_col_set = set()
        for res in eval_results:
            indexes.append(res['model_info']['model_name']+'-'+res['model_info']['agency'])
            dataset_score_list = [(k, main_metric_socre(v)) for k, v in res.items() if k != 'model_info']
            tab_dict = {k: v[1] for k, v in dataset_score_list}
            tab_col_set.update(tab_dict.keys())

            fig_dict = {}
            aver_score = 0.0
            for dataset, score in dataset_score_list:
                if isinstance(score[0], tuple):
                    fig_dict[f'{dataset} I2T'] = score[0][0]
                    fig_dict[f'{dataset} T2I'] = score[0][1]
                    aver_score = aver_score + 0.5 * (score[0][0]+score[0][1])
                else:
                    fig_dict[dataset] = score[0]
                    aver_score = aver_score + score[0]
            fig_col_set.update(fig_dict.keys())

            fig_dict['AVERAGE'] = aver_score
            fig_dict_vals.append(fig_dict)
            tab_dict['AVERAGE'] = aver_score
            tab_dict_vals.append(tab_dict)
        for idx in range(len(indexes)):
            if len(tab_col_set)==0:
                break
            aver_score = fig_dict_vals[idx]['AVERAGE'] * 1.0 / len(tab_col_set)
            fig_dict_vals[idx]['AVERAGE'] = aver_score
            tab_dict_vals[idx]['AVERAGE'] = '%.2f'%aver_score

        tab_cols = ['AVERAGE'] + sorted(tab_col_set)
        fig_cols = ['AVERAGE'] + sorted(fig_col_set)
        tab_df = pd.DataFrame(tab_dict_vals, index=indexes, columns=tab_cols)
        fig_df = pd.DataFrame(fig_dict_vals, index=indexes, columns=fig_cols)
        return tab_df, pd.DataFrame(fig_df.values.T, index=fig_df.columns, columns=fig_df.index)

    def _to_dataframe_single_dataset(self, dataset_name):
        eval_results = self.eval_results
        eval_results = [{
            dataset: metrics for dataset, metrics in res.items()
            if dataset in ['model_info', dataset_name]
        } for res in eval_results]

        indexes = []
        tab_dict_vals = []
        fig_dict_vals = []

        for res in eval_results:
            indexes.append(res['model_info']['model_name'] + '-' + res['model_info']['agency'])
            metrics = res[dataset_name]
            fig_dict_vals.append(metrics)
            tab_dict_vals.append({k: '%.2f'%v for k, v in metrics.items()})
        cols = _DATASET_INFO[dataset_name]['metrics']
        tab_df = pd.DataFrame(tab_dict_vals, index=indexes, columns=cols)
        fig_df = pd.DataFrame(fig_dict_vals, index=indexes, columns=cols)
        return tab_df, pd.DataFrame(fig_df.values.T, index=fig_df.columns, columns=fig_df.index)

    def _to_compare_dataframe(self, df, key1, key2):
        if key1==key2:
            df2 = df[[key1]]
        else:
            df2 = df[[key1, key2]]
        df2['Difference'] = df2[key1]-df2[key2]
        df2 = df2.sort_values(by='Difference', ascending=False)
        return pd.DataFrame({
            'Dataset': df2.index,
            'Difference': df2['Difference']
        })

    def visualize(self):
        st.header('mCLIPEval Evaluation Board')
        table_options = ['Leaderboard Mode', 'Comparison Mode', 'Dataset Mode', 'Customized Mode']

        with st.expander('LEADERBOARD SETTINGS:'):
            selected_langs = st.multiselect('Dataset Languages: ', self.languages, self.languages)
            selected_groups = st.multiselect('Dataset Groups:', self.groups, self.groups)
            selected_tasks = st.multiselect('Dataset Tasks:', self.tasks, self.tasks)
            selected_agencies = st.multiselect('Model Agencies:', self.agencies, self.agencies)
            selected_text_encoders = st.multiselect('Model Text Encoders:', self.text_encoders, self.text_encoders)
            selected_vision_encoders = st.multiselect('Model Vision Encoders:', self.vision_encoders,
                                                      self.vision_encoders)
        df = self._to_dataframe(
            dataset_conditions=[
                ('language', selected_langs),
                ('group', selected_groups),
                ('task', selected_tasks)
            ],
            model_conditions=[
                ('agency', selected_agencies),
                ('vision_encoder', selected_vision_encoders),
                ('text_encoder', selected_text_encoders)
            ]
        )
        tab1, tab2, tab3, tab4 = st.tabs(table_options)
        with tab1:
            st.line_chart(df[1])
            st.dataframe(df[0])

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                model1 = st.selectbox('Choose Model 1:', df[0].index)
            with col2:
                idx = 1 if len(df[0].index)>1 else 0
                model2 = st.selectbox('Choose Model 2:', df[0].index, index=idx)

            source = self._to_compare_dataframe(df[1], model1, model2)
            c = alt.Chart(source).mark_bar().encode(
                x='Difference',
                y=alt.Y('Dataset', sort='-x'),
                color=alt.condition(
                    alt.datum.Difference > 0,
                    alt.value("steelblue"),  # The positive color
                    alt.value("red")  # The negative color
                )
            )
            st.write(f'Performance Comparisons: {model1} vs. {model2}')
            st.altair_chart(c.properties(width=1200))
        with tab3:
            dataset = st.selectbox('Choose Dataset:', [name for name in df[0].columns if name!='AVERAGE'])
            single_df = self._to_dataframe_single_dataset(dataset_name=dataset)
            st.line_chart(single_df[1])
            st.dataframe(single_df[0])

        with tab4:
            full_models = list(df[0].index)
            full_datasets = [name for name in df[0].columns if name!='AVERAGE']
            with st.expander('CLICK TO CUSTOMIZED MODELS AND DATASETS:'):
                selected_models = st.multiselect('Choose Models: ', full_models, full_models)
                selected_datasets = st.multiselect('Choose Datasets: ', full_datasets, full_datasets)
            df_customized = self._to_dataframe(
                dataset_conditions=[
                    ('language', selected_langs),
                    ('group', selected_groups),
                    ('task', selected_tasks)
                ],
                model_conditions=[
                    ('agency', selected_agencies),
                    ('vision_encoder', selected_vision_encoders),
                    ('text_encoder', selected_text_encoders)
                ],
                selected_models = selected_models,
                selected_datasets = selected_datasets
            )
            st.line_chart(df_customized[1])
            st.dataframe(df_customized[0])

