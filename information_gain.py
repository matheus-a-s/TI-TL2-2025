# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. FUNÇÕES DE CÁLCULO DO ALGORITMO ID3 ---

def calculate_entropy(target_column):
    """
    Calcula a entropia de uma coluna (Pandas Series), conforme a fórmula H(S) do artigo.
    A entropia mede a impureza ou incerteza em um conjunto de dados.
    """
    elements, counts = np.unique(target_column, return_counts=True)
    
    # Se o subconjunto é puro (apenas uma classe), a entropia é 0.
    if len(elements) <= 1:
        return 0
    
    # Fórmula da Entropia: -sum(p_i * log2(p_i))
    entropy = np.sum([(-counts[i]/np.sum(counts)) * np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def calculate_information_gain(data, split_attribute_name, target_name="dropout"):
    """
    Calcula o Ganho de Informação para um atributo, conforme a fórmula Gain(T) do artigo.
    Mede a redução na entropia ao dividir os dados por esse atributo.
    """
    # Passo 1: Calcular a entropia total do conjunto de dados (H(S))
    total_entropy = calculate_entropy(data[target_name])
    
    # Encontrar valores únicos e suas contagens para o atributo de divisão
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    
    # Passo 2: Calcular a entropia ponderada após a divisão (Info_T(D))
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * calculate_entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    
    # Passo 3: Ganho de Informação = Entropia Total - Entropia Ponderada
    information_gain = total_entropy - weighted_entropy
    return information_gain

def build_id3_tree(data, original_data, attributes, target_name="dropout", parent_node_class=None):
    """
    Constrói a árvore de decisão de forma recursiva usando o algoritmo ID3.
    """
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

# --- 2. FUNÇÕES DE PLOTAGEM (VERSÃO FINAL CORRIGIDA) ---

# Estilos para os nós da árvore
decision_node_style = dict(boxstyle="round,pad=0.5", fc="#A9CCE3", ec="#2C3E50", lw=2, alpha=0.9)
leaf_node_style_yes = dict(boxstyle="round,pad=0.5", fc="#D5F5E3", ec="#186A3B", lw=2, alpha=0.9)
leaf_node_style_no = dict(boxstyle="round,pad=0.5", fc="#FADBD8", ec="#922B21", lw=2, alpha=0.9)

# Opções de fonte separadas para evitar conflitos
node_font_options = {'fontname': 'Arial', 'fontsize': 10, 'fontweight': 'bold'}
title_font_options = {'fontname': 'Arial', 'fontsize': 20, 'fontweight': 'bold'}

def get_width_and_depth(tree):
    """Calcula a largura e profundidade da árvore para ajudar na plotagem."""
    if not isinstance(tree, dict): return 1, 1
    width, max_depth = 0, 0
    root_attribute = list(tree.keys())[0]
    for value in tree[root_attribute]:
        sub_width, sub_depth = get_width_and_depth(tree[root_attribute][value])
        width += sub_width
        if sub_depth > max_depth: max_depth = sub_depth
    return width, max_depth + 1

def plot_node(ax, node_text, center_pt, node_style):
    """Desenha um nó (caixa de texto) no gráfico."""
    ax.text(center_pt[0], center_pt[1], node_text,
            va="center", ha="center", bbox=node_style, **node_font_options)

def plot_arrow_with_text(ax, parent_pt, child_pt, edge_text):
    """Desenha uma seta curva do nó pai para o filho com texto."""
    arrowprops = dict(arrowstyle="->", color="#34495E", shrinkA=15, shrinkB=15,
                      patchA=None, patchB=None, connectionstyle="arc3,rad=0.15")
    ax.annotate("", xy=parent_pt, xycoords='data',
                xytext=child_pt, textcoords='data', arrowprops=arrowprops)
    
    mid_x = (parent_pt[0] + child_pt[0]) / 2
    mid_y = (parent_pt[1] + child_pt[1]) / 2
    ax.text(mid_x, mid_y, edge_text, va="center", ha="center", 
            rotation=0, fontsize=8, color="#34495E",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))


def plot_tree_recursive(ax, tree, parent_pt, x_offset, y_offset):
    """Função recursiva para desenhar a árvore."""
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
            x_offset += sub_width
        else:
            leaf_style = leaf_node_style_yes if sub_tree == 'Yes' else leaf_node_style_no
            plot_node(ax, sub_tree, (x_child, y_child), leaf_style)
            plot_arrow_with_text(ax, (x_center, y_center), (x_child, y_child), str(value))
            x_offset += sub_width

def plot_tree_with_matplotlib(tree):
    """Função principal para configurar e iniciar a plotagem."""
    fig, ax = plt.subplots(figsize=(14, 9), facecolor='white')
    total_width, total_depth = get_width_and_depth(tree)
    
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, total_depth + 1)
    
    # A chamada recursiva agora não precisa de tantos parâmetros
    plot_tree_recursive(ax, tree, (total_width/2.0, total_depth + 0.5), 0, total_depth - 0.5)
    
    ax.axis('off')
    # CORREÇÃO: Usa apenas o dicionário 'title_font_options' para o estilo do título
    plt.title("Visualização da Árvore de Decisão", y=0.98, **title_font_options)
    plt.show()

# --- 3. EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    try:
        # Passo 1: Carregar os dados
        file_path = r'C:\Users\Lucas Oliveira\Documents\DOUTORADO\TEORIA DA INFORMAÇÃO\DESAFIO\dataset\DataSet_Attendance.csv'
        df = pd.read_csv(file_path, header=0, index_col="index")
        df = df.drop("birth_date", axis=1)
        df = df.drop("valid_final_grade", axis=1)
        # df = df.drop("nacionality", axis=1)
        # df = df.drop("working_student", axis=1)
        # df = df.drop("has_scholarship", axis=1)
        # df = df.drop("moved_student", axis=1)
        # df = df.drop("registrations", axis=1)
        # df = df.drop("academic_year_first_registration", axis=1)
        # df = df.drop("num_registrations", axis=1)
        # df = df.drop("num_registrations_course_conferent_degree", axis=1)
        # df = df.drop("total_ects", axis=1)
        # df = df.drop("attendance_TP", axis=1)
        # df = df.drop("attendance_PL", axis=1)
        # df = df.drop("TP_time", axis=1)
        # df = df.drop("PL_time", axis=1)
        # df = df.drop("normal_grade_result", axis=1)
        # df = df.drop("normal_grade", axis=1)
        # df = df.drop("recovery_grade_result", axis=1)
        # df = df.drop("years_since_registration", axis=1)
        # df = df.drop("age", axis=1)
        # df = df.drop("repetitions", axis=1)
        # df = df.drop("recovery_grade", axis=1)


        # Passo 2: Definir o alvo e obter a lista inicial de todos os atributos
        target_attribute = 'dropout'
        all_attributes = [col for col in df.columns if col != target_attribute]
        
        print(f"Atributo Alvo: '{target_attribute}'")
        print(f"Total de atributos para análise: {len(all_attributes)}\n")

        # NOVO: Passo 3 - Calcular o ganho de informação para todos e armazenar
        print("--- Calculando o Ganho de Informação para todos os atributos ---")
        attribute_gains = {}
        for attr in all_attributes:
            gain = calculate_information_gain(df, attr, target_attribute)
            attribute_gains[attr] = gain
            print(f"Ganho de Informação ({attr}): {gain:.4f}")
        
        # NOVO: Passo 4 - Ordenar e selecionar os 4 melhores atributos
        # Ordena o dicionário pelos valores (ganho) em ordem decrescente
        sorted_attributes = sorted(attribute_gains.items(), key=lambda item: item[1], reverse=True)
        
        # Seleciona os nomes dos 4 melhores
        top_4_attributes = [attr for attr, gain in sorted_attributes[:2]]
        
        print("\n" + "="*60)
        print("OS 4 MELHORES ATRIBUTOS SELECIONADOS (BASEADO NO GANHO DE INFORMAÇÃO):")
        for attr, gain in sorted_attributes[:4]:
            print(f"- {attr} (Ganho: {gain:.4f})")
        print("="*60 + "\n")

        # Passo 5: Construir a árvore de decisão usando APENAS os 4 melhores atributos
        print("Construindo a árvore com os atributos selecionados...")
        decision_tree = build_id3_tree(df, df, top_4_attributes, target_attribute)
        print("\nEstrutura da Árvore de Decisão Gerada:")
        print(decision_tree)
        
        # Passo 6: Visualizar a árvore de decisão
        print("\nGerando a visualização gráfica da árvore...")
        plot_tree_with_matplotlib(decision_tree) # Usando a plotagem horizontal

    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado em '{file_path}'")
        print("Por favor, verifique se o caminho e o nome do arquivo estão corretos.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")
