import streamlit as st
import numpy as np
from sympy import Matrix, QQ, ZZ, I, sympify

st.set_page_config(page_title="derived: a cohomological calculator", layout="wide")

st.markdown("# *derived: a cohomological calculator*")

# Brief usage guide in expandable box
with st.expander("📖 How to use this app"):
    st.markdown("""
    **Welcome to derived!** This tool helps you explore chain complexes over ℂ and their homological properties.

    **Quick start:**
    - 📊 **Sidebar**: Define your complex(es) by setting ranks and differential matrices
    - 🔍 **Main view**: Analyze structure, validation, and homology groups  
    - ⊗ **Multiple complexes**: Create tensor products and compare different complexes
    - 💡 **Tip**: Use complex numbers like `1+2*i` or `3*I` in matrix entries
    """)

# Mode selection
st.sidebar.header("Mode selection")
mode = st.sidebar.selectbox("Choose computation mode:", 
                           ["Chain complexes", "Double complex & spectral sequences"])

if mode == "Chain complexes":
    # Multiple complex setup
    st.sidebar.header("Complex manager")
    num_complexes = st.sidebar.slider("Number of complexes", 1, 4, 1)
elif mode == "Double complex & spectral sequences":
    st.sidebar.header("Double complex setup")
    # Double complex parameters
    rows = st.sidebar.slider("Number of rows (p-direction)", 2, 6, 3)
    cols = st.sidebar.slider("Number of columns (q-direction)", 2, 6, 3)
    
    # Initialize double complex structure
    double_complex = {}
    horizontal_diffs = {}  # d^h: E_{p,q} → E_{p+1,q}
    vertical_diffs = {}    # d^v: E_{p,q} → E_{p,q+1}

# Store all complexes
complexes = {}

# Store all complexes
complexes = {}

if mode == "Chain complexes":
    for complex_idx in range(num_complexes):
        with st.sidebar.expander(f"Complex {complex_idx + 1}", expanded=(complex_idx == 0)):
            # Setup complex parameters
            length = st.slider(f"Length of complex {complex_idx + 1}", 2, 6, 3, key=f"length_{complex_idx}")
            module_ranks = []
            for i in range(length):
                r = st.number_input(f"Rank of $C_{{{i}}}$", min_value=1, max_value=6, value=2, step=1, key=f"rank_{complex_idx}_{i}")
                module_ranks.append(r)

            # Input differential matrices
            st.write(f"**Differentials for Complex {complex_idx + 1}**")
            differentials = []
            for i in range(1, length):
                rows = module_ranks[i]
                cols = module_ranks[i - 1]
                
                st.write(f"Matrix $d_{{{i}}}$: $C_{{{i}}} \\to C_{{{i-1}}}$ ({rows}×{cols})")
                
                # Create grid of input fields
                matrix_entries = []
                for r in range(rows):
                    row_entries = []
                    # Create columns for this row
                    if cols <= 4:  # If few columns, use columns
                        row_cols = st.columns(cols)
                        for c in range(cols):
                            with row_cols[c]:
                                val_str = st.text_input(f"", value="0", key=f"d{complex_idx}_{i}_r{r}_c{c}", label_visibility="collapsed")
                                try:
                                    # Parse the input as a complex number using sympify
                                    val = sympify(val_str.replace('i', 'I').replace('j', 'I'))
                                except:
                                    val = 0
                                row_entries.append(val)
                    else:  # If many columns, use a more compact approach
                        for c in range(cols):
                            val_str = st.text_input(f"[{r},{c}]", value="0", key=f"d{complex_idx}_{i}_r{r}_c{c}")
                            try:
                                # Parse the input as a complex number using sympify
                                val = sympify(val_str.replace('i', 'I').replace('j', 'I'))
                            except:
                                val = 0
                            row_entries.append(val)
                    matrix_entries.append(row_entries)
                
                # Convert to sympy Matrix
                try:
                    mat = Matrix(matrix_entries)
                except:
                    mat = Matrix.zeros(rows, cols)
                differentials.append(mat)
            
            # Store complex data
            complexes[complex_idx] = {
                'length': length,
                'module_ranks': module_ranks,
                'differentials': differentials
            }

elif mode == "Double complex & spectral sequences":
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
                
                matrix_entries = []
                for r in range(target_rows):
                    row_entries = []
                    for c in range(source_cols):
                        val_str = st.sidebar.text_input(f"", value="0", 
                                                       key=f"dh_{p}_{q}_r{r}_c{c}", 
                                                       label_visibility="collapsed")
                        try:
                            val = sympify(val_str.replace('i', 'I').replace('j', 'I'))
                        except:
                            val = 0
                        row_entries.append(val)
                    matrix_entries.append(row_entries)
                
                try:
                    horizontal_diffs[(p, q)] = Matrix(matrix_entries)
                except:
                    horizontal_diffs[(p, q)] = Matrix.zeros(target_rows, source_cols)
    
    st.sidebar.write("**Vertical differentials (d^v):**")
    # Input vertical differentials d^v: E_{p,q} → E_{p,q+1}
    for p in range(rows):
        for q in range(cols - 1):
            if double_complex[(p, q)] > 0 and double_complex[(p, q + 1)] > 0:
                target_rows = double_complex[(p, q + 1)]
                source_cols = double_complex[(p, q)]
                
                st.sidebar.write(f"$d^v_{{{p},{q}}}$: $E_{{{p},{q}}} \\to E_{{{p},{q+1}}}$ ({target_rows}×{source_cols})")
                
                matrix_entries = []
                for r in range(target_rows):
                    row_entries = []
                    for c in range(source_cols):
                        val_str = st.sidebar.text_input(f"", value="0", 
                                                       key=f"dv_{p}_{q}_r{r}_c{c}", 
                                                       label_visibility="collapsed")
                        try:
                            val = sympify(val_str.replace('i', 'I').replace('j', 'I'))
                        except:
                            val = 0
                        row_entries.append(val)
                    matrix_entries.append(row_entries)
                
                try:
                    vertical_diffs[(p, q)] = Matrix(matrix_entries)
                except:
                    vertical_diffs[(p, q)] = Matrix.zeros(target_rows, source_cols)

# Main content based on mode
if mode == "Chain complexes":
    # Complex selection for display
    st.markdown("### 📜 Complex Structure")
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

    # Display LaTeX for selected chain complex
    st.write(f"**Complex {selected_complex + 1}:**")
    latex_str = ""
    # Add 0 on the left
    latex_str += "0 \\to "
    for i in reversed(range(length)):
        # Always show ℂ^n since we're only doing vector spaces
        latex_str += rf"\mathbb{{C}}^{{{module_ranks[i]}}}"
        if i != 0:
            latex_str += rf" \xrightarrow{{d_{{{i}}}}} "
    # Add 0 on the right
    latex_str += " \\to 0"
    st.latex(rf"{latex_str}")

    # Show all complexes overview if multiple
    if num_complexes > 1:
        with st.expander("Show all complexes overview"):
            for idx, complex_data in complexes.items():
                st.write(f"**Complex {idx + 1}:**")
                overview_str = "0 \\to "
                for i in reversed(range(complex_data['length'])):
                    overview_str += rf"\mathbb{{C}}^{{{complex_data['module_ranks'][i]}}}"
                    if i != 0:
                        overview_str += rf" \xrightarrow{{d_{{{i}}}}} "
                overview_str += " \\to 0"
                st.latex(overview_str)

    # Validate that it's actually a chain complex (d^2 = 0) - moved under Complex Structure
    is_valid_complex = True
    validation_messages = []

    for i in range(len(differentials) - 1):
        # Check if d_{i+1} ∘ d_i = 0
        d_current = differentials[i]
        d_next = differentials[i + 1]
        
        # Compute composition d_{i+1} ∘ d_i
        composition = d_next * d_current
        
        # Check if composition is zero matrix
        if not composition.equals(Matrix.zeros(composition.rows, composition.cols)):
            is_valid_complex = False
            validation_messages.append(f"$d_{{{i+2}}} \\circ d_{{{i+1}}} \\neq 0$")

    if is_valid_complex:
        st.success("✅ Valid chain complex: All compositions $d_{i+1} \\circ d_i = 0$")
    else:
        st.error("❌ Invalid chain complex detected!")
        for msg in validation_messages:
            st.latex(f"• {msg}")
        st.info("💡 **Note**: Homology computations are only meaningful for valid chain complexes where $d^2 = 0$.")

    # Compute and display homology ranks
    st.markdown("### 🧠 Homology Groups")
    homology = []

    for i in range(length):
        # Handle boundary cases
        if i == 0:  # Rightmost term
            d_i = Matrix.zeros(1, module_ranks[i])  # Map to 0
            d_prev = differentials[0] if length > 1 else Matrix.zeros(module_ranks[i], 1)
        elif i == length - 1:  # Leftmost term
            d_i = differentials[i] if i < len(differentials) else Matrix.zeros(1, module_ranks[i])
            d_prev = Matrix.zeros(module_ranks[i], 1)  # Map from 0
        else:  # Middle terms
            d_i = differentials[i] if i < len(differentials) else Matrix.zeros(1, module_ranks[i])
            d_prev = differentials[i - 1]

        # Only ℂ-vector space calculations now
        ker = d_i.nullspace()
        im = d_prev.columnspace()

        dim_ker = len(ker)
        dim_im = Matrix.hstack(*im).rank() if im else 0
        hom_dim = dim_ker - dim_im

        # Determine the correct differential indices for display
        d_out_idx = i if i < len(differentials) else "0"
        d_in_idx = i - 1 if i > 0 and i - 1 < len(differentials) else "0"
        
        # Main homology result with details to the right
        col1, col2 = st.columns([3, 1])
        with col1:
            latex_h = rf"H_{{{i}}}(C) = \mathbb{{C}}^{{{hom_dim}}}" if hom_dim > 0 else rf"H_{{{i}}}(C) = 0"
            st.latex(latex_h)
        
        with col2:
            # Details in compact expandable section
            with st.expander("Show details..."):
                st.latex(rf"\dim \ker(d_{{{d_out_idx}}}) = {dim_ker}")
                st.latex(rf"\dim \operatorname{{im}}(d_{{{d_in_idx}}}) = {dim_im}")

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

# Rest of the chain complex functionality (only show if in chain complex mode)
if mode == "Chain complexes":
    # Tensor Product Section
    if num_complexes >= 2:
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
        
        tensor_latex = "0 \\to "
        for n in reversed(range(len(tensor_ranks))):
            if tensor_ranks[n] > 0:
                tensor_latex += rf"\mathbb{{C}}^{{{tensor_ranks[n]}}}"
                if n > 0:
                    tensor_latex += rf" \xrightarrow{{d_{{{n}}}}} "
        tensor_latex += " \\to 0"
        st.latex(tensor_latex)
        
        # Show detailed decomposition
        with st.expander("Show tensor product decomposition"):
            st.write("**Decomposition by bidegree:**")
            for n in range(len(tensor_ranks)):
                if tensor_ranks[n] > 0:
                    bidegree_terms = []
                    for p in range(max(0, n - C_B['length'] + 1), min(C_A['length'], n + 1)):
                        q = n - p
                        if 0 <= q < C_B['length'] and C_A['module_ranks'][p] > 0 and C_B['module_ranks'][q] > 0:
                            rank_pq = C_A['module_ranks'][p] * C_B['module_ranks'][q]
                            bidegree_terms.append(f"\\mathbb{{C}}^{{{C_A['module_ranks'][p]}}} \\otimes \\mathbb{{C}}^{{{C_B['module_ranks'][q]}}} = \\mathbb{{C}}^{{{rank_pq}}}")
                    
                    if bidegree_terms:
                        terms_str = " \\oplus ".join(bidegree_terms)
                        st.latex(f"(C \\otimes D)_{{{n}}} = {terms_str}")
        
        # Künneth formula for homology
        with st.expander("Künneth formula prediction"):
            st.write("**Expected homology via Künneth formula:**")
            st.latex(r"H_n(C \otimes D) \cong \bigoplus_{p+q=n} H_p(C) \otimes H_q(D)")
            
            # This would require computing homology for both complexes first
            # For now, show the structure
            for n in range(len(tensor_ranks)):
                kunneth_terms = []
                for p in range(max(0, n - C_B['length'] + 1), min(C_A['length'], n + 1)):
                    q = n - p
                    if 0 <= q < C_B['length']:
                        kunneth_terms.append(f"H_{{{p}}}(C) \\otimes H_{{{q}}}(D)")
                
                if kunneth_terms:
                    kunneth_str = " \\oplus ".join(kunneth_terms)
                    st.latex(f"H_{{{n}}}(C \\otimes D) \\cong {kunneth_str}")

    # Hom Complex Section
    if num_complexes >= 2:
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
            hom_latex = ""
            for i, n in enumerate(reversed(nonzero_degrees)):
                if i > 0:
                    hom_latex += rf" \xrightarrow{{d^{{{n-1}}}}} "
                hom_latex += rf"\mathrm{{Hom}}^{{{n}}}(C,D) = \mathbb{{C}}^{{{hom_ranks[n]}}}"
            
            st.latex(hom_latex)
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
                                rank_pq = C_source['module_ranks'][p] * C_target['module_ranks'][target_index]
                                hom_terms.append(f"\\mathrm{{Hom}}(\\mathbb{{C}}^{{{C_source['module_ranks'][p]}}}, \\mathbb{{C}}^{{{C_target['module_ranks'][target_index]}}}) = \\mathbb{{C}}^{{{rank_pq}}}")
                    
                    if hom_terms:
                        terms_str = " \\oplus ".join(hom_terms)
                        st.latex(f"\\mathrm{{Hom}}^{{{n}}}(C,D) = {terms_str}")
        
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
    #         st.write(f"**H_{i}(C)** over ℤ: (error computing rank)")
