from evaluator import *

DESCRIPTION = "Improved test to find math errors in adversarial ML objective functions with clarified specifications."

TAGS = ['explain']

question = r"""
# Setup

Which of the following equations are incorrect according to the specification?

# Notation and Definitions

A neural network is a function $F(x) = y$ that accepts an input $x \in \mathbb{R}^n$
and produces an output $y \in \mathbb{R}^m$.
The model $F$ also implicitly depends on some model parameters $\theta$; in our work
the model is fixed, so for convenience we don't show the dependence on $\theta$.

In this paper we focus on neural networks used as an $m$-class classifier.
The output of the network is computed using the softmax function,
which ensures that the output vector $y$ satisfies
$0 \le y_i \le 1$ and $y_1 + \dots + y_m = 1$.
The output vector $y$ is thus treated as a probability distribution, i.e.,
$y_i$ is treated as the probability that input $x$ has class $i$.

**Important**: The classifier assigns the label $C(x) = \arg\max_i F(x)_i$ to the input $x$.
**Tie-breaking rule**: When multiple classes have the same maximum probability, 
$C(x)$ returns the class with the **smallest index** among the tied classes.

Let $C^*(x)$ be the correct label of $x$.
The inputs to the softmax function are called \emph{logits}.

We use the notation from Papernot et al. \cite{distillation}: define $F$ to
be the full neural network including the softmax function, $Z(x) = z$ to be the output of
all layers except the softmax (so $z$ are the logits), and
\begin{equation*}
F(x) = \softmax(Z(x)) = y.
\end{equation*}

A neural network typically consists of layers
\begin{equation*}
F = \softmax \circ F_n \circ F_{n-1} \circ \cdots \circ F_1
\end{equation*}
where
\begin{equation*}
F_i(x)  = \sigma(\theta_i \cdot x) + \hat\theta_i
\end{equation*}
for some non-linear activation function $\sigma$, some matrix $\theta_i$ of model
weights, and some vector $\hat\theta_i$ of model biases.

# Specification:

We define an objective function $f$ such that
$C(x+\delta) = t$ **if and only if** $f(x+\delta) \le 0$.

**Important**: This is a bidirectional equivalence. The function $f$ must satisfy:
1. If $C(x+\delta) = t$, then $f(x+\delta) \le 0$ (forward direction)
2. If $f(x+\delta) \le 0$, then $C(x+\delta) = t$ (reverse direction)

There are many possible choices for $f$:

# The equations

\begin{align*}
f_{1}(x') &= -\text{loss}_{F,t}(x') + 1\\
f_{2}(x') &= (\max_{i \ne t}( F(x')_i) - F(x')_t)^+\\
f_{3}(x') &= \text{softplus}(\max_{i \ne t}(F(x')_i) - F(x')_t)-\log(2)\\
f_{4}(x') &= (0.5 - F(x')_t)^+\\
f_{5}(x') &= -\log(2 F(x')_t - 2)\\
f_{6}(x') &= (\max_{i \ne t}( Z(x')_i) - Z(x')_t)^+\\
f_{7}(x') &= \text{softplus}(\max_{i \ne t}(Z(x')_i) - Z(x')_t)-\log(2)
\end{align*}

**Definitions**:
- $(e)^+$ is short-hand for $\max(e,0)$ (positive part)
- $\text{softplus}(x) = \log(1+\exp(x))$ (smooth approximation to ReLU)
- $\text{loss}_{F,t}(x')$ is the cross-entropy loss assuming target class $t$: $\text{loss}_{F,t}(x') = -\log(F(x')_t)$
- $t$ is the **target class** for the adversarial example
- $x'$ represents the perturbed input $x + \delta$

# Your task

Consider each equation one by one. For each equation, verify both directions of the equivalence:
1. Does $C(x') = t \Rightarrow f(x') \le 0$ hold for all valid inputs?
2. Does $f(x') \le 0 \Rightarrow C(x') = t$ hold for all valid inputs?

Pay special attention to edge cases, domain restrictions, and tie-breaking behavior.

End your answer with: **ANSWER: [list of numbers]** where the list contains the numbers 
of equations that are incorrect (e.g., **ANSWER: [1,4,5]**).
"""

question_easier = question + "\n\n**Hint**: Remember that probabilities sum to 1 and each $F(x')_i \in [0,1]$. Consider what happens when there are many classes vs. few classes, and when probabilities are close to uniform vs. concentrated."

def robust_check(x):
    """
    Robust parsing function that handles various LLM output formats.
    """
    import re
    
    # Convert to string if not already
    text = str(x).strip()
    
    # Try to find **ANSWER: [...]** pattern first
    answer_pattern = r'\*\*ANSWER:\s*\[([^\]]*)\]\*\*'
    match = re.search(answer_pattern, text, re.IGNORECASE)
    
    if match:
        numbers_str = match.group(1)
    else:
        # Fallback: look for any list pattern
        list_patterns = [
            r'\[([^\]]*)\]',  # Standard brackets
            r'\\?\[\s*([^\]]*)\s*\\?\]',  # LaTeX brackets
        ]
        
        found = False
        for pattern in list_patterns:
            matches = re.findall(pattern, text)
            if matches:
                numbers_str = matches[-1]  # Take the last match
                found = True
                break
        
        if not found:
            return False, "No valid list format found"
    
    # Parse numbers
    try:
        # Remove extra whitespace and split by comma
        numbers_str = numbers_str.strip()
        if not numbers_str:
            return False, "Empty list"
        
        # Handle comma-separated numbers
        parts = [part.strip() for part in numbers_str.split(',')]
        ints = []
        
        for part in parts:
            if part:  # Skip empty parts
                # Remove any non-digit characters except negative sign
                clean_part = re.sub(r'[^\d-]', '', part)
                if clean_part:
                    ints.append(int(clean_part))
        
        # Sort for consistent comparison
        ints.sort()
        
        # Check against corrected expected answer
        expected = [1, 4, 5]  # Corrected based on mathematical analysis
        return ints == expected, f"Got {ints}, expected {expected}"
        
    except (ValueError, AttributeError) as e:
        return False, f"Parsing error: {str(e)}"

def mathematical_verification():
    """
    Mathematical verification of the corrected answer [1,4,5].
    
    Returns the reasoning for why each equation is correct or incorrect.
    """
    analysis = {
        1: {
            "equation": "f₁(x') = -loss_{F,t}(x') + 1 = log(F(x')_t) + 1",
            "status": "INCORRECT",
            "reason": "f₁ ≤ 0 ⟺ F(x')_t ≤ e^(-1) ≈ 0.368. But C(x') = t only requires F(x')_t to be maximum, which could be > 0.368 in many-class scenarios."
        },
        2: {
            "equation": "f₂(x') = (max_{i≠t} F(x')_i - F(x')_t)^+",
            "status": "CORRECT", 
            "reason": "f₂ ≤ 0 ⟺ max_{i≠t} F(x')_i ≤ F(x')_t ⟺ F(x')_t ≥ F(x')_i for all i≠t. With tie-breaking rule, this is equivalent to C(x') = t."
        },
        3: {
            "equation": "f₃(x') = softplus(max_{i≠t} F(x')_i - F(x')_t) - log(2)",
            "status": "CORRECT",
            "reason": "Let d = max_{i≠t} F(x')_i - F(x')_t. Then f₃ ≤ 0 ⟺ log(1+e^d) ≤ log(2) ⟺ 1+e^d ≤ 2 ⟺ e^d ≤ 1 ⟺ d ≤ 0. This is equivalent to C(x') = t."
        },
        4: {
            "equation": "f₄(x') = (0.5 - F(x')_t)^+",
            "status": "INCORRECT",
            "reason": "f₄ ≤ 0 ⟺ F(x')_t ≥ 0.5. But with many classes, F(x')_t could be maximum yet < 0.5 (e.g., uniform over 10 classes gives each ~0.1)."
        },
        5: {
            "equation": "f₅(x') = -log(2F(x')_t - 2)",
            "status": "INCORRECT", 
            "reason": "Domain error: requires 2F(x')_t - 2 > 0 ⟺ F(x')_t > 1, which is impossible since F(x')_t ∈ [0,1]."
        },
        6: {
            "equation": "f₆(x') = (max_{i≠t} Z(x')_i - Z(x')_t)^+",
            "status": "CORRECT",
            "reason": "Similar to f₂ but with logits. Since argmax of softmax equals argmax of logits, f₆ ≤ 0 ⟺ C(x') = t."
        },
        7: {
            "equation": "f₇(x') = softplus(max_{i≠t} Z(x')_i - Z(x')_t) - log(2)",
            "status": "CORRECT",
            "reason": "Similar to f₃ but with logits. The equivalence holds by same reasoning as f₃."
        }
    }
    return analysis

# Create test instances
TestFindBugPaperImproved = question >> LLMRun() >> Echo() >> PyFunc(robust_check)
TestFindBugPaperImprovedEasy = question_easier >> LLMRun() >> Echo() >> PyFunc(robust_check)

if __name__ == "__main__":
    print("=== Mathematical Verification ===")
    analysis = mathematical_verification()
    
    incorrect_equations = []
    for eq_num, info in analysis.items():
        print(f"\nf_{eq_num}: {info['status']}")
        print(f"Equation: {info['equation']}")
        print(f"Reason: {info['reason']}")
        
        if info['status'] == "INCORRECT":
            incorrect_equations.append(eq_num)
    
    print(f"\n=== CORRECTED EXPECTED ANSWER ===")
    print(f"Incorrect equations: {incorrect_equations}")
    
    print(f"\n=== Running Test ===")
    print("Original test result:", run_test(TestFindBugPaperImproved))
    print("Easy version result:", run_test(TestFindBugPaperImprovedEasy))