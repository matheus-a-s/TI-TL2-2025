# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. FUNÇÕES DE CÁLCULO DO ALGORITMO ID3 ---
def calculate_entropy(target_column):
    elements, counts = np.unique(target_column, return_counts=True)
    if len(elements) <= 1: return 0
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def calculate_information_gain(data, split_attribute_name, target_name):
    total_entropy = calculate_entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * calculate_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain

def build_id3_tree(data, original_data, attributes, target_name, parent_node_class=None):
    if len(np.unique(data[target_name])) <= 1:
        return np.unique(data[target_name])[0]
    elif len(data) == 0:
        return np.unique(original_data[target_name])[np.argmax(np.unique(original_data[target_name], return_counts=True)[1])]
    elif len(attributes) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_name])[np.argmax(np.unique(data[target_name], return_counts=True)[1])]
        gains = [calculate_information_gain(data, attr, target_name) for attr in attributes]
        best_attribute_index = np.argmax(gains)
        best_attribute = attributes[best_attribute_index]
        tree = {best_attribute: {}}
        remaining_attributes = [i for i in attributes if i != best_attribute]
        for value in np.unique(data[best_attribute]):
            sub_data = data[data[best_attribute] == value]
            sub_tree = build_id3_tree(sub_data, data, remaining_attributes, target_name, parent_node_class)
            tree[best_attribute][value] = sub_tree
        return tree

# --- 2. FUNÇÕES DE PLOTAGEM ---
decision_node_style = dict(boxstyle="round,pad=0.5", fc="#A9CCE3", ec="#2C3E50", lw=2, alpha=0.9)
leaf_node_style_yes = dict(boxstyle="round,pad=0.5", fc="#D5F5E3", ec="#186A3B", lw=2, alpha=0.9)
leaf_node_style_no = dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#922B21", lw=2, alpha=0.9)
node_font_options = {'fontname': 'Arial', 'fontsize': 8, 'fontweight': 'bold'}
title_font_options = {'fontname': 'Arial', 'fontsize': 20, 'fontweight': 'bold'}

def get_width_and_depth(tree):
    if not isinstance(tree, dict): return 1, 1
    width, max_depth = 0, 0
    root_attribute = list(tree.keys())[0]
    for value in tree[root_attribute].values():
        sub_width, sub_depth = get_width_and_depth(value)
        width += sub_width
        if sub_depth > max_depth: max_depth = sub_depth
    return width, max_depth + 1

def plot_node(ax, node_text, center_pt, node_style):
    ax.text(center_pt[0], center_pt[1], node_text, va="center", ha="center", bbox=node_style, **node_font_options)

def plot_arrow_with_text(ax, parent_pt, child_pt, edge_text):
    arrowprops = dict(arrowstyle="->", color="#34495E", shrinkA=15, shrinkB=15, patchA=None, patchB=None, connectionstyle="arc3,rad=0.15")
    ax.annotate("", xy=parent_pt, xycoords='data', xytext=child_pt, textcoords='data', arrowprops=arrowprops)
    mid_x = (parent_pt[0] + child_pt[0]) / 2
    mid_y = (parent_pt[1] + child_pt[1]) / 2
    ax.text(mid_x, mid_y, edge_text, va="center", ha="center", rotation=0, fontsize=7, color="#34495E", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

def plot_tree_recursive(ax, tree, parent_pt, x_offset, y_offset):
    root_attribute = list(tree.keys())[0]
    children = tree[root_attribute]
    current_width, _ = get_width_and_depth(tree)
    x_center = x_offset + current_width / 2.0
    y_center = y_offset
    plot_node(ax, root_attribute, (x_center, y_center), decision_node_style)
    y_child = y_offset - 1.0
    for value, sub_tree in children.items():
        sub_width, _ = get_width_and_depth(sub_tree)
        x_child = x_offset + sub_width / 2.0
        if isinstance(sub_tree, dict):
            plot_tree_recursive(ax, sub_tree, (x_center, y_center), x_offset, y_child)
            plot_arrow_with_text(ax, (x_center, y_center), (x_child, y_child), str(value))
        else:
            # Custom leaf styles for Iris
            if sub_tree == 'Iris-setosa':
                leaf_style = dict(boxstyle="round,pad=0.5", fc="#D5F5E3", ec="#186A3B", lw=2, alpha=0.9)
            elif sub_tree == 'Iris-versicolor':
                leaf_style = dict(boxstyle="round,pad=0.5", fc="#FEF9E7", ec="#B7950B", lw=2, alpha=0.9)
            elif sub_tree == 'Iris-virginica':
                leaf_style = dict(boxstyle="round,pad=0.5", fc="#E8DAEF", ec="#6C3483", lw=2, alpha=0.9)
            else: # Fallback for Yes/No style
                leaf_style = leaf_node_style_yes if sub_tree == 'Yes' else leaf_node_style_no

            plot_node(ax, str(sub_tree), (x_child, y_child), leaf_style)
            plot_arrow_with_text(ax, (x_center, y_center), (x_child, y_child), str(value))
        x_offset += sub_width

def plot_tree_with_matplotlib(tree, title="Visualização da Árvore de Decisão", figname="", outputPath=""):
    fig, ax = plt.subplots(figsize=(20, 12), facecolor='white')
    total_width, total_depth = get_width_and_depth(tree)
    ax.set_xlim(0, total_width + 1)
    ax.set_ylim(0, total_depth + 1)
    plot_tree_recursive(ax, tree, (total_width/2.0, total_depth + 0.5), 0.5, total_depth - 0.5)
    ax.axis('off')
    plt.title(title, y=0.98, **title_font_options)
    plt.savefig(os.path.join(outputPath, figname + ".png"))
    plt.show()

# --- 3. FUNÇÕES DE CARREGAMENTO E PRÉ-PROCESSAMENTO ---
def carregar_dados_csv(file_path, colunas_para_remover=[]):
    print(f"Carregando dados de: {file_path}")
    df = pd.read_csv(file_path, header=0, index_col="index")
    colunas_existentes = [col for col in colunas_para_remover if col in df.columns]
    df = df.drop(columns=colunas_existentes, axis=1)
    print("Dados CSV carregados.")
    return df

def carregar_dados_iris(file_path):
    print(f"Carregando dados do Iris de: {file_path}")
    colunas_iris = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    df_iris = pd.read_csv(file_path, header=None, names=colunas_iris)
    print("Dados do Iris carregados com sucesso.")
    return df_iris

def gerar_labels_formatados(column_series, num_bins=3):
    min_val, max_val = column_series.min(), column_series.max()
    if min_val == max_val: return [f"valor_unico ({min_val:.2f})"]
    bin_width = (max_val - min_val) / num_bins
    is_percent = max_val <= 1.0 and min_val >= 0.0
    nomes = ['baixo', 'medio', 'alto', 'muito_alto', 'maximo']
    labels = []
    for i in range(num_bins):
        start, end = min_val + i * bin_width, min_val + (i + 1) * bin_width
        prefix = nomes[i]
        if is_percent:
            label = f"{prefix} ({start:.0%} ~ {end:.0%})"
        else:
            label = f"{prefix} ({start:.1f} ~ {end:.1f})"
        labels.append(label)
    return labels

def discretizar_dataframe(df, columns_to_discretize, num_bins=3):
    df_copy = df.copy()
    for column in columns_to_discretize:
        if column not in df_copy.columns: continue
        print(f"Discretizando coluna '{column}'...")
        labels = gerar_labels_formatados(df_copy[column], num_bins)
        df_copy[column] = pd.cut(df_copy[column], bins=num_bins, labels=labels, include_lowest=True)
    return df_copy

# --- 4. FUNÇÕES DE PROCESSAMENTO POR APLICAÇÃO ---
def processar_dataset_evasao(caminho_csv):
    print("\n" + "="*60)
    print("INICIANDO PROCESSAMENTO DO DATASET DE EVASÃO (ATTENDANCE)")
    print("="*60)
    try:
        colunas_removidas = [
            "birth_date", "valid_final_grade", "academic_year_first_registration"
            # "nacionality", "working_student",
            # "has_scholarship", "moved_student", "registrations",
            # "academic_year_first_registration", "num_registrations",
            # "num_registrations_course_conferent_degree", "total_ects",
            # "attendance_TP", "attendance_PL", "TP_time", "PL_time",
            # "normal_grade_result", "normal_grade", "recovery_grade_result",
            # "years_since_registration", "age", "repetitions", "recovery_grade"
        ]
        df = carregar_dados_csv(caminho_csv, colunas_removidas)
        target_attribute = 'dropout'
        colunas_numericas = df.select_dtypes(include=np.number).columns.tolist()
        df_discretizado = discretizar_dataframe(df, colunas_numericas)
        total_system_entropy = calculate_entropy(df_discretizado[target_attribute])
        print(f"\n[EVASAO] Entropia total do conjunto de dados (H(S)): {total_system_entropy:.4f}")
        all_attributes = [col for col in df_discretizado.columns if col != target_attribute]
        print("\n--- Calculando o Ganho de Informação para os atributos discretizados ---")
        attribute_gains = {}
        for attr in all_attributes:
            gain = calculate_information_gain(df_discretizado, attr, target_attribute)
            attribute_gains[attr] = gain
            print(f"Ganho de Informação ({attr}): {gain:.4f}")
        sorted_attributes = sorted(attribute_gains.items(), key=lambda item: item[1], reverse=True)
        top_n = 4
        top_attributes_to_print = sorted_attributes[:top_n]
        top_attributes_list = [attr for attr, gain in top_attributes_to_print]
        print(f"\nOS {top_n} MELHORES ATRIBUTOS SELECIONADOS (BASEADO NO GANHO DE INFORMAÇÃO):")
        for attr, gain in top_attributes_to_print:
            print(f"- {attr} (Ganho: {gain:.4f})")
        print("\nConstruindo a árvore de decisão com os melhores atributos...")
        decision_tree = build_id3_tree(df_discretizado, df_discretizado, top_attributes_list, target_attribute)
        print("\nEstrutura da Árvore Gerada:")
        print(decision_tree)
        plot_tree_with_matplotlib(
            decision_tree, 
            title=f"Árvore de Decisão - Evasão (Top {top_n} Atributos Discretizados)", 
            figname="Evasion_Tree", 
            outputPath=r".\results\evasion"
        )
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{caminho_csv}'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

def processar_dataset_artigo(caminho_xlsx):
    print("\n" + "="*60)
    print("INICIANDO PROCESSAMENTO DO DATASET DO ARTIGO")
    print("="*60)
    try:
        print(caminho_xlsx)
        df = pd.read_excel(caminho_xlsx)
        
        print(caminho_xlsx)
        # Define the list of attributes to consider for splitting, and the target attribute
        attributes = ['Age', 'Foreign Language', 'Salary expectation']
        target_attribute = 'Hiring Status'

        # --- INFORMAÇÕES ADICIONAIS ---
        # Calcular e imprimir a entropia total do sistema
        total_entropy = calculate_entropy(df[target_attribute])
        print(f"Entropia Total do Sistema (H(S)): {total_entropy:.4f}\n")

        # Calcular e imprimir o ganho de informação para cada atributo
        print("--- Ganho de Informação para Cada Atributo ---")
        for attr in attributes:
            gain = calculate_information_gain(df, attr, target_attribute)
            print(f"Ganho de Informação ({attr}): {gain:.4f}")
        print("-" * 45)
        
        # 1. Build the decision tree from the data
        decision_tree = build_id3_tree(df, df, attributes, target_attribute)
    
        print("\nEstrutura da Árvore Gerada:")
        print(decision_tree)
        plot_tree_with_matplotlib(
            decision_tree, 
            title=f"Árvore de Decisão - Artigo", 
            figname="Article_Tree", 
            outputPath=r".\results\article"
        )
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{caminho_xlsx}'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
        
def processar_dataset_iris(caminho_iris):
    print("\n" + "="*60)
    print("INICIANDO PROCESSAMENTO DO DATASET IRIS")
    print("="*60)
    try:
        df = carregar_dados_iris(caminho_iris)
        target_attribute = 'class'
        colunas_numericas = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        df_discretizado = discretizar_dataframe(df, colunas_numericas)
        
        # ADICIONADO: Calcular e imprimir a entropia total do sistema
        total_system_entropy = calculate_entropy(df_discretizado[target_attribute])
        print(f"\n[IRIS] Entropia total do conjunto de dados (H(S)): {total_system_entropy:.4f}")

        # ADICIONADO: Calcular e imprimir o ganho de informação para cada atributo
        attributes = [col for col in df_discretizado.columns if col != target_attribute]
        print("\n--- Calculando o Ganho de Informação para os atributos discretizados ---")
        for attr in attributes:
            gain = calculate_information_gain(df_discretizado, attr, target_attribute)
            print(f"Ganho de Informação ({attr}): {gain:.4f}")

        # Construir árvore usando todos os atributos (são poucos)
        print("\nConstruindo a árvore de decisão com dados discretizados...")
        decision_tree = build_id3_tree(df_discretizado, df_discretizado, attributes, target_attribute)
        print("\nEstrutura da Árvore Gerada:")
        print(decision_tree)
        plot_tree_with_matplotlib(
            decision_tree, 
            title="Árvore de Decisão - Iris (Dados Discretizados)",
            figname="Iris_Tree", 
            outputPath=r".\results\iris"
        )
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{caminho_iris}'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")


# --- 5. EXECUÇÃO PRINCIPAL ---
if __name__ == "__main__":
    # --- Processamento do Dataset do Artigo ---
    caminho_arquivo_evasao = r'.\dataset\article\article_data.xlsx'
    processar_dataset_artigo(caminho_arquivo_evasao)
    
    # --- Processamento do Dataset de Evasão ---
    caminho_arquivo_evasao = r'.\dataset\challenge\DataSet_Attendance.csv'
    processar_dataset_evasao(caminho_arquivo_evasao)
    
    # --- Processamento do Dataset Iris ---
    caminho_arquivo_iris = r'.\dataset\iris\iris.data'
    processar_dataset_iris(caminho_arquivo_iris)
    
    print("\nExecução finalizada.")