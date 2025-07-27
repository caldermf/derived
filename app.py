import streamlit as st
from sympy import Matrix, I, sympify

st.set_page_config(page_title="derived: a cohomological calculator", layout="wide")

st.markdown("# *derived: a cohomological calculator*")

# Brief usage guide in expandable box
with st.expander("📖 How to use this app"):
    st.markdown("""
    **Welcome to derived!** This comprehensive tool helps you explore chain complexes over ℂ and their homological properties.

    ## 🎯 **Chain Complexes Mode**
    
    ### **Basic Setup:**
    - 📊 **Sidebar**: Define up to 4 complexes by setting module ranks and differential matrices
    - � **Complex Definition**: Set the length and ranks of each module C^i
    - 🔗 **Differentials**: Input matrices for d^i: C^i → C^{i+1} (cohomological convention)
    - 💡 **Matrix Input**: Use complex numbers like `1+2*i` or `3*I` in entries

    ### **Available Operations:**
    
    **🧠 Cohomology Analysis**
    - Visualize your complex with proper cohomological notation (H^i, d^i, C^i)
    - Automatic validation that d² = 0 (essential for meaningful cohomology)
    - Compute cohomology groups H^i(C) with detailed ker/im dimensions
    - **Shift functionality**: Use [+1]/[-1] buttons to shift degree indexing
    - **Multi-complex view**: Compare multiple complexes side by side

    **⊗ Tensor Product**
    - Compute C ⊗ D for any two complexes (including self-tensor C ⊗ C)
    - See bidegree decomposition and total complex structure
    - **Künneth Formula**: Preview expected homology via H_n(C ⊗ D) ≅ ⊕_{p+q=n} H_p(C) ⊗ H_q(D)

    **🏠 Hom Complex**
    - Build Hom(C,D) complex with proper superscript degrees
    - Compute Ext^n(C,D) groups via cohomology of Hom complex
    - **Endomorphism support**: Analyze Hom(C,C) for self-operations
    - Detailed computation breakdowns with ker/im analysis

    **🏔️ Mapping Cone**
    - Input chain maps f: C → D with automatic validation
    - Construct Cone(f) with block matrix differentials
    - **Long exact sequence**: Understand the induced sequence in homology
    - **Quasi-isomorphism detection**: f induces isomorphism ⟺ Cone(f) is acyclic

    ### **Pro Tips:**
    - ✅ Always ensure your complexes satisfy d² = 0 for meaningful results
    - 📐 The app uses cohomological conventions throughout (superscripts, left-to-right degree increase)
    - 🔄 Self-operations (C ⊗ C, Hom(C,C), f: C → C) are fully supported
    - 🎛️ Use the degree shift feature to align complexes at different starting degrees

    ## 🚧 **Coming Soon**
    - 🎭 **Double Complex & Spectral Sequences**: Currently in development with basic framework implemented
    """)

st.markdown("---")

# Helper functions
def parse_complex_input(val_str):
    """Parse user input as a complex number using sympify."""
    try:
        return sympify(val_str.replace('i', 'I').replace('j', 'I'))
    except:
        return 0

def create_matrix_input_grid(rows, cols, key_prefix):
    """Create a grid of input fields for matrix entry."""
    matrix_entries = []
    for r in range(rows):
        row_entries = []
        if cols <= 4:  # Use columns for few entries
            row_cols = st.columns(cols)
            for c in range(cols):
                with row_cols[c]:
                    val_str = st.text_input(f"Entry [{r},{c}]", value="0", 
                                          key=f"{key_prefix}_r{r}_c{c}", 
                                          label_visibility="collapsed")
                    row_entries.append(parse_complex_input(val_str))
        else:  # Use compact approach for many entries
            for c in range(cols):
                val_str = st.text_input(f"[{r},{c}]", value="0", 
                                      key=f"{key_prefix}_r{r}_c{c}")
                row_entries.append(parse_complex_input(val_str))
        matrix_entries.append(row_entries)
    
    try:
        return Matrix(matrix_entries)
    except:
        return Matrix.zeros(rows, cols)

def build_complex_latex(module_ranks, degree_formula, length, shift=0):
    """Build LaTeX representation of a chain complex."""
    num_cols = 2 + (length * 2) - 1 + 2
    array_latex = f"\\begin{{array}}{{{'c' * num_cols}}}\n"
    
    # First row: the complex with modules and arrows
    complex_elements = ["0", "\\to"]
    for i in range(length):
        complex_elements.append(f"\\mathbb{{C}}^{{{module_ranks[length - 1 - i]}}}")
        if i != length - 1:
            # The differential d_n goes from degree n to degree n-1
            # We're at position i (left to right), which corresponds to degree n
            current_degree = degree_formula(i, length) + shift
            complex_elements.append(f"\\xrightarrow{{d_{{{current_degree}}}}}")
    complex_elements.extend(["\\to", "0"])
    array_latex += " & ".join(complex_elements) + " \\\\\n"
    
    # Second row: degree labels
    degree_elements = ["", ""]  # Empty for "0" and "->"
    for i in range(length):
        degree = degree_formula(i, length) + shift
        degree_elements.append(f"{degree}")
        if i != length - 1:
            degree_elements.append("")  # Empty for arrow
    degree_elements.extend(["", ""])  # Empty for "->" and "0"
    array_latex += " & ".join(degree_elements) + "\n"
    
    array_latex += "\\end{array}"
    return array_latex

def compute_homology_group(degree_index, module_ranks, differentials, display_degree):
    """Compute cohomology group for a given degree, returning also the correct differential labels."""
    length = len(module_ranks)
    
    # Handle boundary cases
    if degree_index == 0:  # Rightmost term
        d_i = Matrix.zeros(1, module_ranks[degree_index])  # Map to 0
        d_prev = differentials[0] if length > 1 else Matrix.zeros(module_ranks[degree_index], 1)
        d_out_label = None  # No outgoing differential to show
        d_in_label = str(display_degree - 1) if length > 1 else None  # Fixed: d^{i-1} comes from degree i-1
    elif degree_index == length - 1:  # Leftmost term
        d_i = differentials[degree_index] if degree_index < len(differentials) else Matrix.zeros(1, module_ranks[degree_index])
        d_prev = Matrix.zeros(module_ranks[degree_index], 1)  # Map from 0
        d_out_label = str(display_degree)  # d^i going out from degree i
        d_in_label = None  # No incoming differential to show
    else:  # Middle terms
        d_i = differentials[degree_index] if degree_index < len(differentials) else Matrix.zeros(1, module_ranks[degree_index])
        d_prev = differentials[degree_index - 1]
        d_out_label = str(display_degree)  # d^i going out from degree i
        d_in_label = str(display_degree - 1)  # d^{i-1} coming from degree i-1

    # Compute cohomology dimension
    ker = d_i.nullspace()
    im = d_prev.columnspace()
    
    dim_ker = len(ker)
    dim_im = Matrix.hstack(*im).rank() if im else 0
    hom_dim = dim_ker - dim_im
    
    return hom_dim, dim_ker, dim_im, d_out_label, d_in_label

def validate_chain_map(chain_map, source_diffs, target_diffs):
    """Validate that a collection of matrices forms a chain map."""
    is_valid = True
    messages = []
    
    for i in range(len(chain_map) - 1):
        if i < len(source_diffs) and i < len(target_diffs):
            try:
                # Check if d_D ∘ f = f ∘ d_C
                left_side = target_diffs[i] * chain_map[i + 1]  # d_D ∘ f_{i+1}
                right_side = chain_map[i] * source_diffs[i]     # f_i ∘ d_C
                
                if not left_side.equals(right_side):
                    is_valid = False
                    messages.append(f"Chain map condition fails at degree {i}: $d_D \\circ f_{{{i+1}}} \\neq f_{{{i}}} \\circ d_C$")
            except:
                messages.append(f"Error validating chain map at degree {i}")
    
    return is_valid, messages

def create_double_complex_diff_input(p, q, source_dim, target_dim, key_prefix):
    """Create input grid for double complex differential."""
    matrix_entries = []
    for r in range(target_dim):
        row_entries = []
        for c in range(source_dim):
            val_str = st.sidebar.text_input(f"Entry [{r},{c}]", value="0", 
                                           key=f"{key_prefix}_{p}_{q}_r{r}_c{c}", 
                                           label_visibility="collapsed")
            row_entries.append(parse_complex_input(val_str))
        matrix_entries.append(row_entries)
    
    try:
        return Matrix(matrix_entries)
    except:
        return Matrix.zeros(target_dim, source_dim)

# Mode selection
st.sidebar.header("Mode selection")
mode = st.sidebar.selectbox("Choose computation mode:", 
                           ["Chain complexes", "Double complex & spectral sequences"])

# Initialize data structures
complexes = {}

if mode == "Chain complexes":
    # Multiple complex setup
    st.sidebar.header("Complex manager")
    num_complexes = st.sidebar.slider("Number of complexes", 1, 4, 1)
    
    # Setup each complex
    for complex_idx in range(num_complexes):
        with st.sidebar.expander(f"Complex {complex_idx + 1}", expanded=(complex_idx == 0)):
            # Setup complex parameters
            length = st.slider(f"Length of complex {complex_idx + 1}", 2, 6, 3, key=f"length_{complex_idx}")
            module_ranks = []
            for i in range(length):
                # Calculate mathematical degree: array position i corresponds to degree (length - 1 - i)
                # But for cohomology, we want to display increasing degrees from left to right
                cohom_degree = i - length + 1  # This gives us the cohomological degree
                r = st.number_input(f"Rank of $C^{{{cohom_degree}}}$", min_value=1, max_value=6, value=2, step=1, key=f"rank_{complex_idx}_{i}")
                module_ranks.append(r)

            # Input differential matrices
            st.write(f"**Differentials for Complex {complex_idx + 1}**")
            differentials = []
            for i in range(1, length):
                rows = module_ranks[i]
                cols = module_ranks[i - 1]
                
                # Calculate the cohomological degree for this differential
                # Array position i-1 corresponds to the source degree, i to target degree
                source_cohom_degree = (i - 1) - length + 1  # Source degree in cohomological notation
                target_cohom_degree = i - length + 1        # Target degree in cohomological notation
                st.write(f"Matrix $d^{{{source_cohom_degree}}}$: $C^{{{source_cohom_degree}}} \\to C^{{{target_cohom_degree}}}$ ({rows}×{cols})")
                mat = create_matrix_input_grid(rows, cols, f"d{complex_idx}_{i}")
                differentials.append(mat)
            
            # Store complex data
            complexes[complex_idx] = {
                'length': length,
                'module_ranks': module_ranks,
                'differentials': differentials
            }

elif mode == "Double complex & spectral sequences":
    st.sidebar.header("Double complex setup")
    # Double complex parameters
    rows = st.sidebar.slider("Number of rows (p-direction)", 2, 6, 3)
    cols = st.sidebar.slider("Number of columns (q-direction)", 2, 6, 3)
    
    # Initialize double complex structure
    double_complex = {}
    horizontal_diffs = {}  # d^h: E_{p,q} → E_{p+1,q}
    vertical_diffs = {}    # d^v: E_{p,q} → E_{p,q+1}
    
    st.sidebar.write("**Double complex entries:**")
    
    # Input ranks for each position (p,q)
    for p in range(rows):
        for q in range(cols):
            rank = st.sidebar.number_input(f"Rank of $E_{{{p},{q}}}$", 
                                         min_value=0, max_value=6, value=1, 
                                         key=f"rank_{p}_{q}")
            double_complex[(p, q)] = rank
    
    st.sidebar.write("**Horizontal differentials (d^h):**")
    # Input horizontal differentials d^h: E_{p,q} → E_{p+1,q}
    for p in range(rows - 1):
        for q in range(cols):
            if double_complex[(p, q)] > 0 and double_complex[(p + 1, q)] > 0:
                target_rows = double_complex[(p + 1, q)]
                source_cols = double_complex[(p, q)]
                
                st.sidebar.write(f"$d^h_{{{p},{q}}}$: $E_{{{p},{q}}} \\to E_{{{p+1},{q}}}$ ({target_rows}×{source_cols})")
                horizontal_diffs[(p, q)] = create_double_complex_diff_input(p, q, source_cols, target_rows, "dh")
    
    st.sidebar.write("**Vertical differentials (d^v):**")
    # Input vertical differentials d^v: E_{p,q} → E_{p,q+1}
    for p in range(rows):
        for q in range(cols - 1):
            if double_complex[(p, q)] > 0 and double_complex[(p, q + 1)] > 0:
                target_rows = double_complex[(p, q + 1)]
                source_cols = double_complex[(p, q)]
                
                st.sidebar.write(f"$d^v_{{{p},{q}}}$: $E_{{{p},{q}}} \\to E_{{{p},{q+1}}}$ ({target_rows}×{source_cols})")
                vertical_diffs[(p, q)] = create_double_complex_diff_input(p, q, source_cols, target_rows, "dv")

# Main content based on mode
if mode == "Chain complexes":
    # Unified feature selector for all operations
    st.markdown("### 🔧 Select an operation")
    feature_options = ["Cohomology Analysis", "Tensor Product", "Hom Complex", "Mapping Cone"] 
    feature_mode = st.selectbox("Select operation:", 
                               feature_options,
                               key="feature_selector")
    
    # Basic Cohomology Analysis (previously the default view)
    if feature_mode == "Cohomology Analysis":
        # Complex selection for display
        if num_complexes > 1:
            selected_complex = st.selectbox("Select complex to analyze:", 
                                           options=list(range(num_complexes)), 
                                           format_func=lambda x: f"Complex {x + 1}")
        else:
            selected_complex = 0

        # Get selected complex data
        current_complex = complexes[selected_complex]
        length = current_complex['length']
        module_ranks = current_complex['module_ranks']
        differentials = current_complex['differentials']

        # Add shift functionality
        if f'shift_{selected_complex}' not in st.session_state:
            st.session_state[f'shift_{selected_complex}'] = 0

        # Display LaTeX for selected chain complex with shift buttons
        col1, col2 = st.columns([4, 1])
        
        with col1:
            st.write(f"**Complex {selected_complex + 1}:**")
            
            # Define degree formula for standard projective resolution ordering
            degree_formula = lambda i, length: i - length + 1
            latex_output = build_complex_latex(module_ranks, degree_formula, length, st.session_state[f'shift_{selected_complex}'])
            st.latex(latex_output)
        
        with col2:
            st.write("**Shift:**")
            shift_col1, shift_col2 = st.columns(2)
            with shift_col1:
                if st.button("[+1]", key=f"shift_down_{selected_complex}"):
                    st.session_state[f'shift_{selected_complex}'] -= 1
                    st.rerun()
            with shift_col2:
                if st.button("[-1]", key=f"shift_up_{selected_complex}"):
                    st.session_state[f'shift_{selected_complex}'] += 1
                    st.rerun()

        # Show all complexes overview if multiple
        if num_complexes > 1:
            with st.expander("Show all complexes overview"):
                for idx, complex_data in complexes.items():
                    st.write(f"**Complex {idx + 1}:**")
                    latex_output = build_complex_latex(complex_data['module_ranks'], degree_formula, complex_data['length'])
                    st.latex(latex_output)

        # Validate that it's actually a chain complex (d^2 = 0)
        is_valid_complex = True
        validation_messages = []

        for i in range(len(differentials) - 1):
            # Check if d_{i+1} ∘ d_i = 0
            composition = differentials[i + 1] * differentials[i]
            
            if not composition.equals(Matrix.zeros(composition.rows, composition.cols)):
                is_valid_complex = False
                # Calculate the correct mathematical degree labels for the error message
                # Array index i corresponds to differential at degree (length - 1 - i) in our left-to-right ordering
                degree_i = (length - 1 - i) + st.session_state[f'shift_{selected_complex}']
                degree_i_plus_1 = degree_i + 1
                validation_messages.append(f"$d_{{{degree_i}}} \\circ d_{{{degree_i_plus_1}}} \\neq 0$")

        if is_valid_complex:
            st.success("✅ Valid chain complex: All compositions $d_{i+1} \\circ d_i = 0$")
        else:
            st.error("❌ Invalid chain complex detected!")
            for msg in validation_messages:
                st.latex(f"• {msg}")
            st.info("💡 **Note**: Cohomology computations are only meaningful for valid chain complexes where $d^2 = 0$.")

        # Compute and display cohomology ranks
        st.markdown("### 🧠 Cohomology Groups")

        for i in range(length):
            # Map display position to degree index in data structure
            degree_index = length - 1 - i
            display_degree = i - length + 1 + st.session_state[f'shift_{selected_complex}']  # Include shift in display degree
            
            hom_dim, dim_ker, dim_im, d_out_label, d_in_label = compute_homology_group(degree_index, module_ranks, differentials, display_degree)
            
            # Main cohomology result with details to the right
            col1, col2 = st.columns([3, 1])
            with col1:
                latex_h = rf"H^{{{display_degree}}}(C) = \mathbb{{C}}^{{{hom_dim}}}" if hom_dim > 0 else rf"H^{{{display_degree}}}(C) = 0"
                st.latex(latex_h)
            
            with col2:
                # Details in compact expandable section
                with st.expander("Show details..."):
                    if d_out_label is not None:
                        st.latex(rf"\dim \ker(d^{{{d_out_label}}}) = {dim_ker}")
                    if d_in_label is not None:
                        st.latex(rf"\dim \operatorname{{im}}(d^{{{d_in_label}}}) = {dim_im}")

    # Tensor Product Section
    elif feature_mode == "Tensor Product":
        st.markdown("### ⊗ Tensor Product")
        
        # Select two complexes for tensor product
        col1, col2 = st.columns(2)
        with col1:
            complex_A = st.selectbox("Select first complex:", 
                                    options=list(range(num_complexes)), 
                                    format_func=lambda x: f"Complex {x + 1}",
                                    key="tensor_A")
        with col2:
            complex_B = st.selectbox("Select second complex:", 
                                    options=list(range(num_complexes)), 
                                    format_func=lambda x: f"Complex {x + 1}",
                                    key="tensor_B")
        
        # Show info about self-tensor if same complex selected
        if complex_A == complex_B:
            st.info("💡 Computing C ⊗ C (tensor product of complex with itself). You can add more complexes too using the sidebar!")
        
        # Get the two complexes
        C_A = complexes[complex_A]
        C_B = complexes[complex_B]
        
        # Compute tensor product complex
        max_degree = C_A['length'] + C_B['length'] - 2
        tensor_ranks = []
        tensor_differentials = []
        
        # Compute ranks for each degree n = p + q
        for n in range(max_degree + 1):
            total_rank = 0
            for p in range(max(0, n - C_B['length'] + 1), min(C_A['length'], n + 1)):
                q = n - p
                if 0 <= q < C_B['length']:
                    total_rank += C_A['module_ranks'][p] * C_B['module_ranks'][q]
            tensor_ranks.append(total_rank)
        
        # Display tensor product complex
        st.write(f"**Tensor Product: Complex {complex_A + 1} ⊗ Complex {complex_B + 1}:**")
        
        # Build properly aligned complex with degree labels using multi-column array
        # Count non-zero terms for proper column calculation
        nonzero_terms = [n for n in range(len(tensor_ranks)) if tensor_ranks[n] > 0]
        if nonzero_terms:
            num_cols = 2 + (len(nonzero_terms) * 2) - 1 + 2
            array_latex = f"\\begin{{array}}{{{'c' * num_cols}}}\n"
            
            # First row: the complex
            complex_elements = ["0", "\\to"]
            for i, n in enumerate(nonzero_terms):
                complex_elements.append(f"\\mathbb{{C}}^{{{tensor_ranks[n]}}}")
                if i < len(nonzero_terms) - 1:  # Not the last element
                    complex_elements.append(f"\\xrightarrow{{d_{{{n}}}}}")
            complex_elements.extend(["\\to", "0"])
            array_latex += " & ".join(complex_elements) + " \\\\\n"
            
            # Second row: degree labels
            degree_elements = ["", ""]  # Empty for "0" and "->"
            for i, n in enumerate(nonzero_terms):
                degree_elements.append(f"{n}")
                if i < len(nonzero_terms) - 1:  # Not the last element
                    degree_elements.append("")  # Empty for arrow
            degree_elements.extend(["", ""])  # Empty for "->" and "0"
            array_latex += " & ".join(degree_elements) + "\n"
            
            array_latex += "\\end{array}"
            st.latex(array_latex)
        else:
            st.latex("0")
        
        # Show detailed decomposition
        with st.expander("Show tensor product decomposition"):
            st.write("**Decomposition by total degree:**")
            for n in range(len(tensor_ranks)):
                if tensor_ranks[n] > 0:
                    bidegree_terms = []
                    for p in range(max(0, n - C_B['length'] + 1), min(C_A['length'], n + 1)):
                        q = n - p
                        if 0 <= q < C_B['length'] and C_A['module_ranks'][p] > 0 and C_B['module_ranks'][q] > 0:
                            # Convert array indices to cohomological degrees (left-to-right ordering)
                            cohom_degree_p = p - C_A['length'] + 1
                            cohom_degree_q = q - C_B['length'] + 1
                            rank_pq = C_A['module_ranks'][p] * C_B['module_ranks'][q]
                            bidegree_terms.append(f"C^{{{cohom_degree_p}}} \\otimes D^{{{cohom_degree_q}}}")
                    
                    if bidegree_terms:
                        # Convert total degree to cohomological indexing
                        total_cohom_degree = n - (C_A['length'] + C_B['length'] - 2)
                        terms_str = " \\oplus ".join(bidegree_terms)
                        st.latex(f"(C \\otimes D)^{{{total_cohom_degree}}} = {terms_str}")
                        
                        # Show dimension calculation separately
                        st.write(f"   Dimension: {tensor_ranks[n]}")
        
        # Künneth formula for homology
        #with st.expander("Künneth formula prediction"):
        ##    st.write("**Expected homology via Künneth formula:**")
        #    st.latex(r"H_n(C \otimes D) \cong \bigoplus_{p+q=n} H_p(C) \otimes H_q(D)")
            
            # This would require computing homology for both complexes first
            # For now, show the structure
        #    for n in range(len(tensor_ranks)):
        #        kunneth_terms = []
        #        for p in range(max(0, n - C_B['length'] + 1), min(C_A['length'], n + 1)):
        #            q = n - p
        #            if 0 <= q < C_B['length']:
        #                kunneth_terms.append(f"H_{{{p}}}(C) \\otimes H_{{{q}}}(D)")
        #        
        #        if kunneth_terms:
        #            kunneth_str = " \\oplus ".join(kunneth_terms)
        #            st.latex(f"H_{{{n}}}(C \\otimes D) \\cong {kunneth_str}")

    # Hom Complex Section
    elif feature_mode == "Hom Complex":
        st.markdown("### 🏠 Hom Complex")
        
        # Select two complexes for Hom
        col1, col2 = st.columns(2)
        with col1:
            complex_C = st.selectbox("Select source complex (C):", 
                                    options=list(range(num_complexes)), 
                                    format_func=lambda x: f"Complex {x + 1}",
                                    key="hom_C")
        with col2:
            complex_D = st.selectbox("Select target complex (D):", 
                                    options=list(range(num_complexes)), 
                                    format_func=lambda x: f"Complex {x + 1}",
                                    key="hom_D")
        
        # Show info about self-Hom if same complex selected
        if complex_C == complex_D:
            st.info("💡 Computing Hom(C, C) (endomorphism complex). You can add more complexes too using the sidebar!")
        
        # Get the two complexes
        C_source = complexes[complex_C]
        C_target = complexes[complex_D]
        
        # Compute Hom complex: Hom(C, D)^n = ∏_{p} Hom(C_p, D_{p+n})
        max_hom_degree = C_target['length'] - 1
        min_hom_degree = -(C_source['length'] - 1)
        hom_ranks = {}
        hom_differentials = {}
        
        # Helper function to construct block matrices for Hom differentials
        def construct_hom_differential(n):
            """Construct the differential d^n: Hom^n(C,D) → Hom^{n+1}(C,D)"""
            if n + 1 not in hom_ranks or hom_ranks[n] == 0 or hom_ranks[n + 1] == 0:
                return Matrix.zeros(hom_ranks.get(n + 1, 0), hom_ranks.get(n, 0))
            
            # Build block matrix structure
            blocks = []
            target_block_row = 0
            
            for p_target in range(C_source['length']):
                target_idx_out = p_target + (n + 1)  # For Hom^{n+1}
                if 0 <= target_idx_out < C_target['length']:
                    block_row = []
                    source_block_col = 0
                    
                    for p_source in range(C_source['length']):
                        target_idx_in = p_source + n  # For Hom^n
                        if 0 <= target_idx_in < C_target['length']:
                            # This block represents the map from Hom(C_p_source, D_{p_source+n}) to Hom(C_p_target, D_{p_target+n+1})
                            block_rows = C_source['module_ranks'][p_target] * C_target['module_ranks'][target_idx_out]
                            block_cols = C_source['module_ranks'][p_source] * C_target['module_ranks'][target_idx_in]
                            
                            if p_target == p_source:
                                # Diagonal block: d_D ∘ f term
                                if target_idx_out - 1 < len(C_target['differentials']):
                                    d_D = C_target['differentials'][target_idx_out - 1]  # d: D_{p+n+1} → D_{p+n}
                                    # Kronecker product: I ⊗ d_D^T (since we're in Hom space)
                                    I_C = Matrix.eye(C_source['module_ranks'][p_source])
                                    block = I_C.tensor_product(d_D.T)
                                else:
                                    block = Matrix.zeros(block_rows, block_cols)
                            elif p_target == p_source + 1:
                                # Super-diagonal block: (-1)^n f ∘ d_C term
                                if p_source < len(C_source['differentials']):
                                    d_C = C_source['differentials'][p_source]  # d: C_{p+1} → C_p
                                    I_D = Matrix.eye(C_target['module_ranks'][target_idx_in])
                                    sign = (-1) ** n
                                    block = sign * d_C.T.tensor_product(I_D)
                                else:
                                    block = Matrix.zeros(block_rows, block_cols)
                            else:
                                block = Matrix.zeros(block_rows, block_cols)
                            
                            block_row.append(block)
                            source_block_col += block_cols
                        
                    if block_row:
                        blocks.append(Matrix.hstack(*block_row))
                    target_block_row += block_rows
            
            if blocks:
                return Matrix.vstack(*blocks)
            else:
                return Matrix.zeros(hom_ranks.get(n + 1, 0), hom_ranks.get(n, 0))
        
        # Compute ranks for each degree n
        for n in range(min_hom_degree, max_hom_degree + 1):
            total_rank = 0
            for p in range(C_source['length']):
                target_index = p + n
                if 0 <= target_index < C_target['length']:
                    # Hom(ℂ^a, ℂ^b) ≅ ℂ^{a*b}
                    total_rank += C_source['module_ranks'][p] * C_target['module_ranks'][target_index]
            hom_ranks[n] = total_rank
        
        # Compute differentials (only for non-zero adjacent degrees)
        for n in sorted(hom_ranks.keys()):
            if n + 1 in hom_ranks and hom_ranks[n] > 0 and hom_ranks[n + 1] > 0:
                try:
                    hom_differentials[n] = construct_hom_differential(n)
                except:
                    # Fallback to zero matrix if construction fails
                    hom_differentials[n] = Matrix.zeros(hom_ranks[n + 1], hom_ranks[n])
        
        # Display Hom complex
        st.write(f"**Hom Complex: Hom(Complex {complex_C + 1}, Complex {complex_D + 1}):**")
        
        # Build LaTeX string for non-zero terms
        nonzero_degrees = [n for n in sorted(hom_ranks.keys()) if hom_ranks[n] > 0]
        if nonzero_degrees:
            # Build properly aligned complex with degree labels using multi-column array
            num_cols = 2 + (len(nonzero_degrees) * 2) - 1 + 2  # Add space for leading/trailing zeros
            array_latex = f"\\begin{{array}}{{{'c' * num_cols}}}\n"
            
            # First row: the complex with modules and arrows
            complex_elements = ["0", "\\to"]
            for i, n in enumerate(nonzero_degrees):
                complex_elements.append(f"\\mathbb{{C}}^{{{hom_ranks[n]}}}")
                if i < len(nonzero_degrees) - 1:  # Not the last element
                    complex_elements.append(f"\\xrightarrow{{d^{{{n}}}}}")
            complex_elements.extend(["\\to", "0"])
            array_latex += " & ".join(complex_elements) + " \\\\\n"
            
            # Second row: degree labels
            degree_elements = ["", ""]  # Empty for "0" and "->"
            for i, n in enumerate(nonzero_degrees):
                degree_elements.append(f"{n}")
                if i < len(nonzero_degrees) - 1:  # Not the last element
                    degree_elements.append("")  # Empty for arrow
            degree_elements.extend(["", ""])  # Empty for "->" and "0"
            array_latex += " & ".join(degree_elements) + "\n"
            
            array_latex += "\\end{array}"
            st.latex(array_latex)
        else:
            st.latex(r"\mathrm{Hom}(C,D) = 0")
        
        # Show detailed decomposition
        with st.expander("Show Hom complex decomposition"):
            st.write("**Decomposition by degree:**")
            for n in sorted(hom_ranks.keys()):
                if hom_ranks[n] > 0:
                    hom_terms = []
                    for p in range(C_source['length']):
                        target_index = p + n
                        if 0 <= target_index < C_target['length']:
                            if C_source['module_ranks'][p] > 0 and C_target['module_ranks'][target_index] > 0:
                                # Convert array indices to cohomological degrees
                                cohom_degree_p = p - C_source['length'] + 1
                                cohom_degree_q = target_index - C_target['length'] + 1
                                hom_terms.append(f"\\mathrm{{Hom}}(C^{{{cohom_degree_p}}}, D^{{{cohom_degree_q}}})")
                    
                    if hom_terms:
                        terms_str = " \\oplus ".join(hom_terms)
                        st.latex(f"\\mathrm{{Hom}}^{{{n}}}(C,D) = {terms_str}")
                        
                        # Show dimension calculation separately
                        st.write(f"   Dimension: {hom_ranks[n]}")
        
        # Show expected Ext groups
        with st.expander("Ext groups computation"):
            st.write("**Computed Ext groups via Hom complex cohomology:**")
            st.latex(r"\mathrm{Ext}^n(C,D) = H^n(\mathrm{Hom}(C,D))")
            
            # Compute cohomology of the Hom complex
            ext_groups = {}
            for n in sorted(hom_ranks.keys()):
                if hom_ranks[n] > 0:
                    # Get the differentials going in and out of degree n
                    d_in = hom_differentials.get(n - 1, Matrix.zeros(hom_ranks[n], hom_ranks.get(n - 1, 0)))
                    d_out = hom_differentials.get(n, Matrix.zeros(hom_ranks.get(n + 1, 0), hom_ranks[n]))
                    
                    try:
                        # Compute kernel and image
                        ker = d_out.nullspace() if d_out.rows > 0 and d_out.cols > 0 else []
                        im = d_in.columnspace() if d_in.rows > 0 and d_in.cols > 0 else []
                        
                        dim_ker = len(ker)
                        dim_im = Matrix.hstack(*im).rank() if im else 0
                        ext_dim = dim_ker - dim_im
                        
                        ext_groups[n] = max(0, ext_dim)
                        
                        # Display result
                        if ext_groups[n] > 0:
                            st.latex(f"\\mathrm{{Ext}}^{{{n}}}(C,D) = \\mathbb{{C}}^{{{ext_groups[n]}}}")
                        else:
                            st.latex(f"\\mathrm{{Ext}}^{{{n}}}(C,D) = 0")
                            
                    except Exception as e:
                        st.latex(f"\\mathrm{{Ext}}^{{{n}}}(C,D) = \\text{{(computation error)}}")
            
            # Show computation details
            with st.expander("Show Ext computation details"):
                for n in sorted(ext_groups.keys()):
                    if hom_ranks[n] > 0:
                        d_in = hom_differentials.get(n - 1, Matrix.zeros(hom_ranks[n], hom_ranks.get(n - 1, 0)))
                        d_out = hom_differentials.get(n, Matrix.zeros(hom_ranks.get(n + 1, 0), hom_ranks[n]))
                        
                        st.write(f"**Degree {n}:**")
                        st.write(f"- Hom space dimension: {hom_ranks[n]}")
                        st.write(f"- Incoming differential dimension: {d_in.rows}×{d_in.cols}")
                        st.write(f"- Outgoing differential dimension: {d_out.rows}×{d_out.cols}")
                        
                        try:
                            ker = d_out.nullspace() if d_out.rows > 0 and d_out.cols > 0 else []
                            im = d_in.columnspace() if d_in.rows > 0 and d_in.cols > 0 else []
                            
                            dim_ker = len(ker)
                            dim_im = Matrix.hstack(*im).rank() if im else 0
                            
                            st.latex(f"\\dim \\ker(d^{{{n}}}) = {dim_ker}")
                            st.latex(f"\\dim \\operatorname{{im}}(d^{{{n-1}}}) = {dim_im}")
                            st.latex(f"\\dim \\mathrm{{Ext}}^{{{n}}}(C,D) = {dim_ker} - {dim_im} = {max(0, dim_ker - dim_im)}")
                        except:
                            st.write("Error in detailed computation")

    # Mapping Cone Section
    elif feature_mode == "Mapping Cone":
        st.markdown("### 🏔️ Mapping Cone")
        
        # Select two complexes for mapping cone
        col1, col2 = st.columns(2)
        with col1:
            source_complex = st.selectbox("Select source complex (C):", 
                                        options=list(range(num_complexes)), 
                                        format_func=lambda x: f"Complex {x + 1}",
                                        key="cone_source")
        with col2:
            target_complex = st.selectbox("Select target complex (D):", 
                                        options=list(range(num_complexes)), 
                                        format_func=lambda x: f"Complex {x + 1}",
                                        key="cone_target")
        
        # Show info about self-mapping cone if same complex selected
        if source_complex == target_complex:
            st.info("💡 Computing mapping cone of f: C → C (endomorphism). You can add more complexes too using the sidebar!")
        
        # Get the two complexes
        C_source = complexes[source_complex]
        C_target = complexes[target_complex]
        
        # Input chain map f: C → D
        st.write(f"**Chain map f: Complex {source_complex + 1} → Complex {target_complex + 1}:**")
        
        # Determine compatible length (minimum of the two)
        compatible_length = min(C_source['length'], C_target['length'])
        
        # Input matrices for each degree of the chain map
        chain_map = []
        for i in range(compatible_length):
            source_rank = C_source['module_ranks'][i]
            target_rank = C_target['module_ranks'][i]
            
            # Calculate mathematical degree: array position i corresponds to degree (length - 1 - i)
            # But for chain maps, we want to use a consistent degree system
            math_degree = compatible_length - 1 - i
            st.write(f"Matrix $f_{{{math_degree}}}$: $C_{{{math_degree}}} \\to D_{{{math_degree}}}$ ({target_rank}×{source_rank})")
            
            # Use helper function for matrix input
            mat = create_matrix_input_grid(target_rank, source_rank, f"f{source_complex}_{target_complex}_{i}")
            chain_map.append(mat)
        
        # Validate that f is indeed a chain map (f ∘ d_C = d_D ∘ f)
        is_valid_chain_map, chain_map_validation_messages = validate_chain_map(chain_map, C_source['differentials'], C_target['differentials'])
        
        if is_valid_chain_map:
            st.success("✅ Valid chain map: All squares commute")
        else:
            st.error("❌ Invalid chain map detected!")
            for msg in chain_map_validation_messages:
                st.latex(f"• {msg}")
            st.info("💡 **Note**: Mapping cone is only well-defined for valid chain maps.")
        
        # Compute mapping cone: Cone(f)_n = C_{n-1} ⊕ D_n
        st.write("**Mapping Cone: Cone(f):**")
        
        # Calculate cone complex structure
        cone_length = max(C_source['length'], C_target['length']) + 1
        cone_ranks = []
        cone_differentials = []
        
        # Compute ranks: Cone(f)_n = C_{n-1} ⊕ D_n
        for n in range(cone_length):
            c_rank = C_source['module_ranks'][n - 1] if 0 <= n - 1 < C_source['length'] else 0
            d_rank = C_target['module_ranks'][n] if 0 <= n < C_target['length'] else 0
            cone_ranks.append(c_rank + d_rank)
        
        # Compute differentials: d_cone = [d_C, f; 0, d_D] in block form
        for n in range(1, cone_length):
            if cone_ranks[n] > 0 and cone_ranks[n - 1] > 0:
                # Get dimensions for block matrix
                c_n_rank = C_source['module_ranks'][n - 1] if 0 <= n - 1 < C_source['length'] else 0
                d_n_rank = C_target['module_ranks'][n] if 0 <= n < C_target['length'] else 0
                c_n1_rank = C_source['module_ranks'][n - 2] if 0 <= n - 2 < C_source['length'] else 0
                d_n1_rank = C_target['module_ranks'][n - 1] if 0 <= n - 1 < C_target['length'] else 0
                
                # Build block matrix differential
                blocks = []
                
                # Top row: [d_C, f]
                top_blocks = []
                
                # d_C block (if exists)
                if c_n_rank > 0 and c_n1_rank > 0 and n - 2 < len(C_source['differentials']):
                    d_C_block = C_source['differentials'][n - 2]  # d: C_{n-1} → C_{n-2}
                else:
                    d_C_block = Matrix.zeros(c_n1_rank, c_n_rank)
                top_blocks.append(d_C_block)
                
                # f block (with sign (-1)^n)
                if d_n1_rank > 0 and c_n_rank > 0 and n - 1 < len(chain_map):
                    f_block = ((-1) ** n) * chain_map[n - 1]  # f: C_{n-1} → D_{n-1}
                else:
                    f_block = Matrix.zeros(d_n1_rank, c_n_rank)
                top_blocks.append(f_block)
                
                # Bottom row: [0, d_D]
                bottom_blocks = []
                
                # 0 block
                zero_block = Matrix.zeros(d_n1_rank, c_n_rank)
                bottom_blocks.append(zero_block)
                
                # d_D block (if exists)
                if d_n_rank > 0 and d_n1_rank > 0 and n - 1 < len(C_target['differentials']):
                    d_D_block = C_target['differentials'][n - 1]  # d: D_n → D_{n-1}
                else:
                    d_D_block = Matrix.zeros(d_n1_rank, d_n_rank)
                bottom_blocks.append(d_D_block)
                
                # Combine blocks
                try:
                    if top_blocks[0].cols > 0 or top_blocks[1].cols > 0:
                        top_row = Matrix.hstack(*top_blocks) if len(top_blocks) > 1 else top_blocks[0]
                    else:
                        top_row = Matrix.zeros(c_n1_rank + d_n1_rank, 0)
                        
                    if bottom_blocks[0].cols > 0 or bottom_blocks[1].cols > 0:
                        bottom_row = Matrix.hstack(*bottom_blocks) if len(bottom_blocks) > 1 else bottom_blocks[0]
                    else:
                        bottom_row = Matrix.zeros(c_n1_rank + d_n1_rank, 0)
                    
                    if top_row.rows > 0 and bottom_row.rows > 0:
                        cone_diff = Matrix.vstack(top_row, bottom_row)
                    else:
                        cone_diff = Matrix.zeros(cone_ranks[n - 1], cone_ranks[n])
                        
                    cone_differentials.append(cone_diff)
                except:
                    # Fallback to zero matrix
                    cone_differentials.append(Matrix.zeros(cone_ranks[n - 1], cone_ranks[n]))
            else:
                cone_differentials.append(Matrix.zeros(cone_ranks[n - 1] if n > 0 else 0, cone_ranks[n]))
        
        # Display mapping cone complex
        nonzero_cone_terms = [n for n in range(len(cone_ranks)) if cone_ranks[n] > 0]
        if nonzero_cone_terms:
            # Build properly aligned complex with degree labels using multi-column array
            num_cols = 2 + (len(nonzero_cone_terms) * 2) - 1 + 2
            array_latex = f"\\begin{{array}}{{{'c' * num_cols}}}\n"
            
            # First row: the complex
            complex_elements = ["0", "\\to"]
            for i, n in enumerate(nonzero_cone_terms):
                complex_elements.append(f"\\mathbb{{C}}^{{{cone_ranks[n]}}}")
                if i < len(nonzero_cone_terms) - 1:  # Not the last element
                    complex_elements.append(f"\\xrightarrow{{d_{{{n}}}}}")
            complex_elements.extend(["\\to", "0"])
            array_latex += " & ".join(complex_elements) + " \\\\\n"
            
            # Second row: degree labels
            degree_elements = ["", ""]  # Empty for "0" and "->"
            for i, n in enumerate(nonzero_cone_terms):
                degree_elements.append(f"{n}")
                if i < len(nonzero_cone_terms) - 1:  # Not the last element
                    degree_elements.append("")  # Empty for arrow
            degree_elements.extend(["", ""])  # Empty for "->" and "0"
            array_latex += " & ".join(degree_elements) + "\n"
            
            array_latex += "\\end{array}"
            st.latex(array_latex)
        else:
            st.latex(r"\text{Cone}(f) = 0")
        
        # Show detailed decomposition
        with st.expander("Show mapping cone decomposition"):
            st.write("**Decomposition at each degree:**")
            for n in range(len(cone_ranks)):
                if cone_ranks[n] > 0:
                    c_rank = C_source['module_ranks'][n - 1] if 0 <= n - 1 < C_source['length'] else 0
                    d_rank = C_target['module_ranks'][n] if 0 <= n < C_target['length'] else 0
                    
                    terms = []
                    if c_rank > 0:
                        terms.append(f"C_{{{n-1}}} = \\mathbb{{C}}^{{{c_rank}}}")
                    if d_rank > 0:
                        terms.append(f"D_{{{n}}} = \\mathbb{{C}}^{{{d_rank}}}")
                    
                    if terms:
                        terms_str = " \\oplus ".join(terms)
                        st.latex(f"\\text{{Cone}}(f)_{{{n}}} = {terms_str}")
        
        # Show differential structure
        with st.expander("Show mapping cone differential structure"):
            st.write("**Differential matrices:**")
            st.latex(r"d_{\text{cone}} = \begin{pmatrix} d_C & (-1)^n f \\ 0 & d_D \end{pmatrix}")
            
            for n in range(1, min(len(cone_differentials) + 1, 4)):  # Show first few differentials
                if n - 1 < len(cone_differentials) and cone_ranks[n] > 0 and cone_ranks[n - 1] > 0:
                    st.write(f"**Differential d_{n}:**")
                    
                    # Show the block structure
                    c_n_rank = C_source['module_ranks'][n - 1] if 0 <= n - 1 < C_source['length'] else 0
                    d_n_rank = C_target['module_ranks'][n] if 0 <= n < C_target['length'] else 0
                    c_n1_rank = C_source['module_ranks'][n - 2] if 0 <= n - 2 < C_source['length'] else 0
                    d_n1_rank = C_target['module_ranks'][n - 1] if 0 <= n - 1 < C_target['length'] else 0
                    
                    block_info = f"Block structure: ({c_n1_rank + d_n1_rank}) × ({c_n_rank + d_n_rank})"
                    st.write(block_info)
                    
                    if c_n_rank > 0 and c_n1_rank > 0:
                        st.write(f"- Top-left: d_C ({c_n1_rank} × {c_n_rank})")
                    if d_n1_rank > 0 and c_n_rank > 0:
                        st.write(f"- Top-right: (-1)^{n} f ({d_n1_rank} × {c_n_rank})")
                    if d_n_rank > 0 and d_n1_rank > 0:
                        st.write(f"- Bottom-right: d_D ({d_n1_rank} × {d_n_rank})")
                    st.write(f"- Bottom-left: 0 ({d_n1_rank} × {c_n_rank})")
        
        # Long exact sequence in homology
        with st.expander("Long exact sequence in homology"):
            st.write("**Long exact sequence induced by the mapping cone:**")
            st.latex(r"\cdots \to H_{n+1}(\text{Cone}(f)) \to H_n(C) \xrightarrow{f_*} H_n(D) \to H_n(\text{Cone}(f)) \to H_{n-1}(C) \to \cdots")
            st.write("The mapping cone fits into this long exact sequence, where f₊ is the map induced by f on homology.")
            st.info("💡 **Key property**: f is a quasi-isomorphism (induces isomorphism on homology) if and only if Cone(f) is acyclic (has zero homology).")

elif mode == "Double complex & spectral sequences":
    st.markdown("### 🎭 Double Complex & Spectral Sequences")
    
    # Display double complex structure
    st.write("**Double Complex Structure:**")
    
    # Create a visual diagram using arrays and arrow symbols
    st.write("**Bigraded structure E_{p,q} with differentials:**")
    
    # Build a grid using arrays with arrows
    # First, create the basic grid
    grid_latex = "\\begin{array}{" + "c" * (2 * rows - 1) + "}\n"
    
    # Build rows from top to bottom (q decreasing)
    for q_idx, q in enumerate(reversed(range(cols))):
        # Add the modules row
        row_elements = []
        for p in range(rows):
            rank = double_complex.get((p, q), 0)
            if rank > 0:
                element = f"\\mathbb{{C}}^{{{rank}}}"
            else:
                element = "0"
            
            row_elements.append(element)
            
            # Add horizontal arrow if not the last column and differential exists
            if (p < rows - 1 and double_complex.get((p, q), 0) > 0 and 
                double_complex.get((p+1, q), 0) > 0 and
                horizontal_diffs.get((p, q)) is not None):
                row_elements.append("\\xrightarrow{d^h}")
            elif p < rows - 1:
                row_elements.append("\\phantom{\\xrightarrow{d^h}}")  # Invisible spacer
        
        grid_latex += " & ".join(row_elements) + " \\\\\n"
        
        # Add vertical arrows row if not the last row
        if q > 0:
            arrow_row = []
            for p in range(rows):
                # Check if vertical differential exists
                if (double_complex.get((p, q), 0) > 0 and double_complex.get((p, q-1), 0) > 0 and
                    vertical_diffs.get((p, q-1)) is not None):
                    arrow_row.append("\\downarrow_{d^v}")
                else:
                    arrow_row.append("\\phantom{\\downarrow}")
                
                # Add spacer for horizontal arrow position
                if p < rows - 1:
                    arrow_row.append("\\phantom{\\xrightarrow{d^h}}")
            
            grid_latex += " & ".join(arrow_row) + " \\\\\n"
    
    grid_latex += "\\end{array}"
    st.latex(grid_latex)
    
    # Validate double complex (commutative squares)
    st.markdown("### ✅ Double Complex Validation")
    is_valid_double = True
    double_validation_messages = []
    
    # Check that all squares commute: d^v ∘ d^h + d^h ∘ d^v = 0
    for p in range(rows - 1):
        for q in range(cols - 1):
            # Get the four relevant differentials
            dh_pq = horizontal_diffs.get((p, q), Matrix.zeros(double_complex.get((p+1, q), 0), double_complex.get((p, q), 0)))
            dv_p1q = vertical_diffs.get((p+1, q), Matrix.zeros(double_complex.get((p+1, q+1), 0), double_complex.get((p+1, q), 0)))
            dv_pq = vertical_diffs.get((p, q), Matrix.zeros(double_complex.get((p, q+1), 0), double_complex.get((p, q), 0)))
            dh_pq1 = horizontal_diffs.get((p, q+1), Matrix.zeros(double_complex.get((p+1, q+1), 0), double_complex.get((p, q+1), 0)))
            
            # Check if the square commutes: d^v_{p+1,q} ∘ d^h_{p,q} + d^h_{p,q+1} ∘ d^v_{p,q} = 0
            if (dh_pq.rows > 0 and dh_pq.cols > 0 and dv_p1q.rows > 0 and dv_p1q.cols > 0 and
                dv_pq.rows > 0 and dv_pq.cols > 0 and dh_pq1.rows > 0 and dh_pq1.cols > 0):
                try:
                    composition1 = dv_p1q * dh_pq
                    composition2 = dh_pq1 * dv_pq
                    anticommutator = composition1 + composition2
                    
                    if not anticommutator.equals(Matrix.zeros(anticommutator.rows, anticommutator.cols)):
                        is_valid_double = False
                        double_validation_messages.append(f"Square at ({p},{q}) does not commute")
                except:
                    # If matrices can't be multiplied, skip validation
                    pass
    
    if is_valid_double:
        st.success("✅ Valid double complex: All squares commute")
    else:
        st.error("❌ Invalid double complex detected!")
        for msg in double_validation_messages:
            st.write(f"• {msg}")
    
    st.markdown("### 📊 Spectral Sequence Computation")
    
    # Spectral sequence page selector
    max_page = min(rows, cols) + 2  # Reasonable upper bound
    current_page = st.slider("Spectral sequence page (r)", 0, max_page, 0)
    
    if current_page == 0:
        st.write("**E^0 page (original double complex):**")
        st.write("E^0_{p,q} = E_{p,q}")
        st.write("The diagram above shows the E^0 page with horizontal and vertical differentials.")
        
    elif current_page == 1:
        st.write("**E^1 page (horizontal cohomology):**")
        
        # Compute spectral sequence page iteratively
        E_current = {}
        
        # Initialize E^1 page (cohomology with respect to horizontal differential)
        st.write("Computing E^1 page (horizontal cohomology)...")
        
        # For each q, compute cohomology of the horizontal complex
        for q in range(cols):
            # Build horizontal complex for fixed q
            horizontal_complex = []
            horizontal_diffs_q = []
            
            for p in range(rows):
                horizontal_complex.append(double_complex.get((p, q), 0))
            
            for p in range(rows - 1):
                horizontal_diffs_q.append(horizontal_diffs.get((p, q), Matrix.zeros(double_complex.get((p+1, q), 0), double_complex.get((p, q), 0))))
            
            # Compute cohomology for each position
            for p in range(rows):
                if horizontal_complex[p] > 0:
                    # Get differentials going in and out
                    d_in = horizontal_diffs_q[p-1] if p > 0 else Matrix.zeros(horizontal_complex[p], 0)
                    d_out = horizontal_diffs_q[p] if p < len(horizontal_diffs_q) else Matrix.zeros(0, horizontal_complex[p])
                    
                    try:
                        # Compute cohomology dimension
                        ker = d_out.nullspace() if d_out.rows > 0 and d_out.cols > 0 else []
                        im = d_in.columnspace() if d_in.rows > 0 and d_in.cols > 0 else []
                        
                        dim_ker = len(ker)
                        dim_im = Matrix.hstack(*im).rank() if im else 0
                        cohomology_dim = max(0, dim_ker - dim_im)
                        
                        E_current[(p, q)] = cohomology_dim
                    except:
                        E_current[(p, q)] = 0
                else:
                    E_current[(p, q)] = 0
        
        # Build diagram for E^1 page with vertical differentials
        st.write("**E^1_{p,q} with vertical differentials d^1: E^1_{p,q} → E^1_{p,q-1}:**")
        
        # Create array-based diagram
        grid_latex = "\\begin{array}{" + "c" * rows + "}\n"
        
        # Build rows from top to bottom (q decreasing)
        for q_idx, q in enumerate(reversed(range(cols))):
            # Add the modules row
            row_elements = []
            for p in range(rows):
                rank = E_current.get((p, q), 0)
                if rank > 0:
                    row_elements.append(f"\\mathbb{{C}}^{{{rank}}}")
                else:
                    row_elements.append("0")
            
            grid_latex += " & ".join(row_elements) + " \\\\\n"
            
            # Add vertical arrows row if not the last row
            if q > 0:
                arrow_row = []
                for p in range(rows):
                    # Check if both source and target are non-zero for vertical differential
                    if (E_current.get((p, q), 0) > 0 and E_current.get((p, q-1), 0) > 0):
                        arrow_row.append("\\downarrow_{d^1}")
                    else:
                        arrow_row.append("\\phantom{\\downarrow}")
                
                grid_latex += " & ".join(arrow_row) + " \\\\\n"
        
        grid_latex += "\\end{array}"
        st.latex(grid_latex)
    
    elif current_page >= 2:
        st.write(f"**E^{current_page} page:**")
        
        # Build diagram showing the diagonal differentials
        st.write(f"**E^{current_page}_{{p,q}} with diagonal differentials d^{current_page}: E^{current_page}_{{p,q}} → E^{current_page}_{{p+{current_page},q-{current_page-1}}}:**")
        
        # Create a simplified representation showing the differential pattern
        st.write("**Differential pattern:**")
        
        # Show a few examples of the diagonal arrows
        examples = []
        for q in reversed(range(min(cols, 3))):  # Show only first few for clarity
            for p in range(min(rows, 3)):
                target_p = p + current_page
                target_q = q - (current_page - 1)
                
                if (target_p < rows and target_q >= 0 and target_q < cols):
                    examples.append(f"E^{{{current_page}}}_{{{p},{q}}} \\xrightarrow{{d^{{{current_page}}}}} E^{{{current_page}}}_{{{target_p},{target_q}}}")
        
        if examples:
            for example in examples[:6]:  # Show first 6 examples
                st.latex(example)
            if len(examples) > 6:
                st.write("...")
        else:
            st.write("No differentials exist for this page in the current grid size.")
        
        # Show the general structure as a grid
        st.write("**Grid structure:**")
        grid_latex = "\\begin{array}{" + "c" * rows + "}\n"
        
        for q in reversed(range(cols)):
            row_elements = []
            for p in range(rows):
                element = f"E^{{{current_page}}}_{{{p},{q}}}"
                
                # Add diagonal arrow notation if target exists
                target_p = p + current_page
                target_q = q - (current_page - 1)
                
                if (target_p < rows and target_q >= 0 and target_q < cols):
                    element += f"^{{\\rightarrow({target_p},{target_q})}}"
                
                row_elements.append(element)
            
            grid_latex += " & ".join(row_elements) + " \\\\\n"
        
        grid_latex += "\\end{array}"
        st.latex(grid_latex)
        
        st.info(f"💡 Computing the actual E^{current_page} terms requires implementing the spectral sequence differential d^{current_page-1} and taking cohomology. The diagram shows the structure of differentials on this page.")
    
    # Convergence information
    with st.expander("Spectral sequence convergence"):
        st.write("**Convergence of the spectral sequence:**")
        st.latex(r"E^r_{p,q} \Rightarrow H^{p+q}(\text{Tot}(E))")
        st.write("The spectral sequence converges to the cohomology of the total complex associated to the double complex.")
        st.write("For finite double complexes, the spectral sequence stabilizes at a finite page r₀, giving E^∞_{p,q} = E^{r₀}_{p,q}.")