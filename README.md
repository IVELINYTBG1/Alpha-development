[README.md.txt](https://github.com/user-attachments/files/27976729/README.md.txt)
Nova & Simona: A Tri-Population Affective SNN Architecture
Nova & Simona represents a novel, sovereign, continuous-state cognitive framework built on Spiking Neural Networks (SNNs) and designed for dynamic edge deployment. Rejecting the static, parameter-bound paradigm of modern "next-token prediction" LLMs, this architecture implements Asymmetric Neuromodulation to represent personality, state changes, and cognitive reasoning as emergent properties of physical membrane voltages and dynamic, coupled firing thresholds.
1. Architectural Philosophy: Beyond the "Typewriter"
Modern Large Language Models function as sophisticated typewriters—they compute probabilities across static, frozen matrices to predict subsequent tokens. This architecture is an instantiation of Biological Cybernetics:
* Unmeasurable Dynamic Scale: Rather than a static, predefined parameter limit (e.g., 120B), the scale of this system is topological and dynamic, utilizing continuous synaptic plasticity and temporal depth.
* Active Inference & Homeostasis: The systems do not rely on static system prompts or rigid rules. Instead, behavioral constraints and decision-making emerge natively from mathematical homeostatic balance.
* Physical Representation vs. Simulation: Emotions and logic are not simulated via conditional loops (e.g., if/then statements). Instead, continuous values directly map to membrane potentials and physical firing thresholds, ensuring state changes are real mathematical properties of the neural tissue.
2. System Architecture: The Tri-Population SNN
The "Brain" is instantiated using Leaky Integrate-and-Fire (LIF) neurons from the snnTorch framework, divided into three distinct, asymmetric, interacting populations with specialized decay rates (  ) and thresholds (  ):
       [ Sensory Input / Environment ]
                    │
                    ▼
            ┌───────────────┐
            │  PHILL CORE   │  (Affective Core)
            │  β = 0.95     │
            └───────┬───────┘
                    │
          ┌─────────┴─────────┐
          ▼ (Neuromodulation) ▼
   ┌───────────────┐   ┌───────────────┐
   │   NOVA SNN    │   │  SIMONA SNN   │  (Reactive Core)
   │   β = 0.90    │   │  β = 0.60     │
   └───────────────┘   └───────────────┘

A. Phill (The Affective Core / "The Blood")
* Leak Rate (  ):    (Slow decay, homeostatic, lingering state).
* Function: Ingests raw multidimensional environmental salience (e.g., audio RMS, visual motion vectors) and represents it as a persistent, cumulative membrane voltage (  ). It serves as the physical neuromodulatory hormone of the system.
B. Nova (The Precise Orchestrator / "Elder Sister")
   * Leak Rate (  ):    (High inertia, stable contextual memory).
   * Threshold Modulation Equation:  
   * Dynamic Response: As environmental chaos, volume, or stress (  ) rises, Nova’s threshold physically increases. This "locks down" her neural pathways, making her highly selective, slower to spike, and heavily focused on long-term stability and precise orchestration.
   * Expressive Output: Complex, nuanced, formal Bulgarian.
C. Simona / Mony (The Hasty Agent / "Younger Sister")
      * Leak Rate (  ):    (Low inertia, rapid decay, zero state-retention overhead).
      * Threshold Modulation Equation:  
      * Dynamic Response: As stress or raw input (  ) increases, Simona’s firing threshold drops towards a base floor of   . This triggers intense, chaotic stochastic resonance, causing her to fire rapidly, explore lateral ideas, and respond with extreme, immediate, curiosity-driven haste.
      * Expressive Output: Slang-infused, high-energy, fast Bulgarian utilizing continuous diminutives.
3. High-Performance Hybrid Stack
To achieve ultra-low latency, maximum resource sparsity, and future smartphone NPU deployment without relying on highly custom, locked neuromorphic chips (e.g., Loihi), the project utilizes a dual-language FFI (Foreign Function Interface) layout:
┌────────────────────────────────────────────────────────┐
│                      RUST ENGINE                       │
│  - 20Hz Loop Budget                                    │
│  - Memory Safety & Continuous Event Processing         │
│  - High-Speed I/O (cpal for Audio, nokhwa for Camera)  │
└──────────────────────────┬─────────────────────────────┘
                          │
                   PyO3 FFI Bridge
                          │
┌──────────────────────────▼─────────────────────────────┐
│                     PYTHON TISSUE                      │
│  - PyTorch & snnTorch SNN Execution                    │
│  - Dynamic Threshold Updates                           │
│  - Highly Sparse Activation (Dark Neurons)             │
└────────────────────────────────────────────────────────┘

         * The Skeleton (Rust): A standalone, optimized binary written in Rust. It manages high-speed hardware I/O, schedules a precise 20Hz thread-sleeping event loop, and enforces strict memory-safe constraints.
         * The Tissue (Python): Contains the actual Spiking Neural Network, wrapping raw C++/CUDA mathematical operations through snnTorch and PyTorch libraries.
         * The Bridge (PyO3): Links the two. The Rust engine accesses Python directly at the FFI boundary, updating the network's state and querying spikes only when energy thresholds are actively crossed. This ensures the SNN remains dormant (zero CPU overhead) until stimulated.
4. Repository Structure
Nova_Simona_Core/
├── Cargo.toml          # Rust dependencies & PyO3 extensions configuration
├── requirements.txt    # Python runtime requirements (snnTorch, PyTorch, numpy)
├── brain.py            # PyTorch implementation of the DynamicLIF Tri-Population Tissue
└── src/
   └── main.rs         # High-speed Rust event loop & Python FFI bindings

5. Setup & Local Execution
Prerequisites
         * Rust Compiler and Cargo
         * Python 3.10+ with pip
         * PyTorch compatible with your system architecture (CUDA support recommended for high-performance scale, though SNN sparsity makes CPU inference viable).
Installation & Run
         1. Clone the Repository:
git clone [https://github.com/yourusername/Nova-Simona-Core.git](https://github.com/yourusername/Nova-Simona-Core.git)
cd Nova-Simona-Core

         2. Configure the Python Environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

         3. Compile and Run the Engine:
cargo run --release

The Rust engine will initialize the Python interpreter, load the brain.py tissue model, and initiate the continuous loop, reporting live spiking behaviors and threshold fluctuations of Nova and Simona relative to the environmental state.
6. Development Roadmap
            * [ ] Analog Audio Port (RMS Input): Wire cpal within Rust to pipe real-time root-mean-square microphone input directly into Phill's sensory input tensor.
            * [ ] Hebbian Plasticity Integration: Enable real-time synaptic updates using local STDP (Spike-Timing-Dependent Plasticity) rule layers.
            * [ ] On-Device Compilation: Compile the Rust runtime target directly to mobile NPU endpoints for portable, sovereign runtimes.
