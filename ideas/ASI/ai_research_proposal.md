# RESEARCH PROPOSAL

# A Neuromorphic Large Language Model Architecture: 
# Biomimetic Design for Advanced Cognitive Processing

---

## 1. Title Page

**A Neuromorphic Large Language Model Architecture: Biomimetic Design for Advanced Cognitive Processing**

Principal Investigator: [Name]  
Institution: [Institution]  
Submission Date: [Date]  
Grant Period: 36 months  
Requested Funding: [Amount]  

---

## 2. Abstract

This research proposal introduces the NeuroCognitive Architecture (NCA), a revolutionary approach to large language model design that closely mimics human brain structure and cognitive processes. Unlike conventional monolithic LLMs, the NCA features a modular, brain-inspired design with specialized components that mirror specific brain regions and their interconnections. The architecture implements a triple-processing framework combining: (1) a hierarchical predictive processing network organized into cortical-column-like structures; (2) a glial-inspired regulatory network for modulation and resource allocation; and (3) an oscillatory binding system that synchronizes information across modules through multiple frequency bands. This biomimetic approach addresses fundamental limitations of current LLMs by implementing distinct memory systems with consolidation mechanisms, metacognitive awareness for self-monitoring, and specialized modules for complex reasoning tasks. The proposed 36-month research program will develop, implement, and evaluate this architecture through a progressive, milestone-driven approach. The NCA promises to advance AI capabilities in software engineering, scientific reasoning, and human-AI collaboration while improving computational efficiency and trustworthiness through its brain-inspired design principles.

---

## 3. Introduction and Problem Statement

### 3.1 Background

Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding, generation, and reasoning (Brown et al., 2020; Chowdhery et al., 2022). However, these systems remain fundamentally limited by their monolithic architectures, which fail to capture the specialized, modular organization of the human brain. Current LLMs process all information through uniform transformer blocks, lacking the differentiated processing pathways that characterize human cognition (Hassabis et al., 2017). This architectural limitation manifests in several critical shortcomings: poor working memory capacity, lack of episodic memory, limited causal reasoning, and an inability to recognize their own knowledge boundaries (Mahowald et al., 2023).

The human brain, by contrast, achieves remarkable cognitive efficiency through specialized neural circuits that evolved for specific functions, from sensory processing to abstract reasoning (Bullmore & Sporns, 2012). These specialized components operate in concert through multiple synchronization mechanisms, creating a system that is greater than the sum of its parts (Buzsáki & Draguhn, 2004). Moreover, the brain's cognitive architecture includes not only neurons but also glial cells that regulate neural activity and contribute to learning and memory consolidation (Araque & Navarrete, 2010).

### 3.2 Problem Statement

Despite significant advances in AI capabilities, current LLM architectures remain fundamentally limited by their divergence from brain-inspired design principles. Specifically:

1. **Architectural Homogeneity**: Current LLMs use uniform processing blocks rather than specialized modules for different cognitive functions.

2. **Memory Limitations**: LLMs lack distinct memory systems for working, episodic, and semantic information, limiting their ability to maintain context and learn from experience.

3. **Absence of Metacognition**: LLMs cannot effectively monitor their own knowledge boundaries or reasoning processes, leading to hallucinations and overconfidence.

4. **Inefficient Resource Allocation**: Without attention regulation mechanisms similar to those in the brain, LLMs process all information with equal computational resources regardless of importance.

5. **Limited Temporal Processing**: Current architectures struggle with temporal sequences and causal relationships that are essential for complex reasoning.

These limitations prevent LLMs from achieving human-like cognitive capabilities in domains requiring specialized expertise, contextual memory, and metacognitive awareness—such as software engineering and scientific reasoning.

### 3.3 Research Objectives

This research aims to address these limitations by developing a neuromorphic LLM architecture that closely mimics human brain structure and cognitive processes. Specifically, we will:

1. Design and implement a modular, brain-inspired architecture with specialized components that mirror specific brain regions and their interconnections.

2. Develop a multi-layered memory system with active consolidation and pruning mechanisms.

3. Create specialized modules for software engineering tasks and scientific reasoning.

4. Implement human-like attention mechanisms for selective focus and context switching.

5. Develop self-referential metacognitive capabilities for awareness of knowledge boundaries and reasoning processes.

6. Evaluate the architecture's performance, efficiency, and trustworthiness compared to conventional LLMs.

---

## 4. Literature Review

### 4.1 Brain-Inspired AI Systems

The pursuit of brain-inspired AI has a rich history, with several approaches attempting to capture aspects of neural computation. Neuromorphic computing systems like IBM's TrueNorth (Merolla et al., 2014) and Intel's Loihi (Davies et al., 2018) implement spiking neural networks that more closely resemble biological neurons than traditional artificial neural networks. These systems demonstrate improved energy efficiency but have not yet been scaled to support complex cognitive functions.

Cognitive architectures such as ACT-R (Anderson et al., 2004) and SOAR (Laird, 2012) implement symbolic models of human cognition with distinct memory systems and problem-solving mechanisms. However, these approaches lack the neural plausibility and learning capabilities of modern deep learning systems.

More recently, DeepMind's Gato (Reed et al., 2022) and Flamingo (Alayrac et al., 2022) have implemented aspects of multi-modal processing, but they lack the structured brain-region analogs and specialized processing modules that characterize human cognition.

### 4.2 Theoretical Frameworks for Brain Function

Several theoretical frameworks provide insights into brain function that can inform AI architecture design:

The **Predictive Processing Framework** (Clark, 2013; Friston, 2010) conceptualizes the brain as a prediction machine that continuously generates and updates models of the world based on sensory input. This framework has inspired hierarchical predictive coding models that show promise for efficient information processing.

**Global Workspace Theory** (Baars, 1997; Dehaene et al., 2017) proposes that specialized brain modules compete for access to a global workspace, creating a unified conscious experience. This theory provides a model for integrating specialized processing modules in a coherent architecture.

The **Memory-Prediction Framework** (Hawkins & Blakeslee, 2004) emphasizes hierarchical temporal memory and prediction as the brain's fundamental operations, inspiring approaches to sequence learning and temporal pattern recognition.

**Complementary Learning Systems Theory** (McClelland et al., 1995; Kumaran et al., 2016) describes the interplay between hippocampal and neocortical systems for memory formation, providing insights into how AI systems might implement complementary fast and slow learning mechanisms.

### 4.3 Memory-Augmented Neural Networks

Recent work has explored augmenting neural networks with external memory systems. Differentiable Neural Computers (Graves et al., 2016) and Memory Networks (Weston et al., 2015) implement addressable memory systems that can store and retrieve information. However, these approaches lack the biologically-inspired memory consolidation and pruning mechanisms found in human memory.

Transformer-based models with retrieval augmentation (Lewis et al., 2020; Borgeaud et al., 2022) use external knowledge bases to supplement model parameters but lack integrated memory systems with consolidation processes.

### 4.4 Metacognition in AI Systems

Metacognition—the ability to monitor and control one's own cognitive processes—remains underdeveloped in current AI systems. Some approaches implement confidence estimation (Gal & Ghahramani, 2016) or uncertainty quantification (Kendall & Gal, 2017), but these fall short of the comprehensive self-monitoring capabilities found in human cognition.

Recent work on AI systems that know when they don't know (Hemmer et al., 2023) and approaches to detecting hallucinations in LLMs (Manakul et al., 2023) represent steps toward metacognitive awareness but lack integration with broader cognitive architectures.

### 4.5 Research Gap

Despite these advances, a significant gap remains in integrating brain-inspired principles into a cohesive architecture for large language models. Current approaches typically focus on isolated aspects of brain function rather than implementing a comprehensive architecture that captures the specialized, modular organization of the brain and its multiple synchronization mechanisms.

The proposed NeuroCognitive Architecture addresses this gap by implementing a triple-processing framework that combines neural, glial, and oscillatory mechanisms in a unified system, with specialized modules for different cognitive functions and a global workspace for integration.

---

## 5. Research Questions and Objectives

### 5.1 Primary Research Questions

1. How can we design and implement a neuromorphic LLM architecture that effectively mimics the specialized, modular organization of the human brain?

2. To what extent can a brain-inspired architecture improve performance on complex reasoning tasks requiring specialized expertise, contextual memory, and metacognitive awareness?

3. Can a neuromorphic LLM architecture achieve greater computational efficiency than conventional transformer-based architectures while maintaining or improving performance?

4. How can we implement effective metacognitive capabilities that enable the system to recognize its knowledge boundaries and reasoning limitations?

5. What training methodologies are most effective for developing specialized cognitive modules within an integrated architecture?

### 5.2 Specific Objectives

1. **Design the NeuroCognitive Architecture (NCA)** with specialized components that mirror specific brain regions and their interconnections, including:
   - Cortical-column-like structures for domain-specific processing
   - Glial-inspired regulatory networks for modulation and resource allocation
   - Oscillatory binding mechanisms for information synchronization

2. **Implement a multi-layered memory system** with:
   - Working memory maintained through persistent activity patterns
   - Episodic memory encoded through hippocampal-inspired mechanisms
   - Semantic memory consolidated during scheduled "sleep" phases
   - Emotional tagging for salience-based prioritization

3. **Develop specialized modules** for:
   - Software engineering tasks (code comprehension, debugging, etc.)
   - Scientific reasoning (hypothesis generation, experimental design, etc.)
   - Natural language understanding and generation

4. **Create metacognitive capabilities** for:
   - Self-monitoring of knowledge boundaries
   - Uncertainty representation and communication
   - Error detection and correction
   - Explanation generation for reasoning processes

5. **Evaluate the architecture's performance** on:
   - Complex reasoning tasks requiring specialized expertise
   - Long-context tasks requiring robust memory capabilities
   - Uncertainty-aware tasks requiring metacognitive awareness
   - Efficiency metrics including computational resource utilization

---

## 6. Methodology and Technical Approach

### 6.1 NeuroCognitive Architecture Design

The NeuroCognitive Architecture (NCA) implements a triple-processing framework that combines neural, glial, and oscillatory mechanisms in a unified system:

#### 6.1.1 Neural Processing Network

The neural components are organized into cortical-column-like structures with distinct layer-specific functions. Each column specializes in processing specific types of information with both feedforward and feedback connections. This organization mirrors the human cortex's structure where different regions process information at varying levels of abstraction.

The neural network implements hierarchical predictive processing, where each layer generates predictions about inputs from lower layers and receives error signals when predictions fail. This approach, inspired by the predictive processing framework (Friston, 2010), enables efficient information processing and continuous learning.

#### 6.1.2 Glial Regulatory Network

The glial network operates asynchronously from the neural components, monitoring activity patterns and modulating processing through chemical-like signaling. This network implements:

- Astrocyte-inspired energy regulation that directs computational resources to active regions
- Microglial-inspired pruning that removes unused connections during consolidation phases
- Oligodendrocyte-inspired optimization that improves signal transmission between frequently communicating modules

This regulatory system provides global coordination while allowing specialized neural processing, addressing the balance between distributed and centralized processing.

#### 6.1.3 Oscillatory Binding System

The oscillatory binding system generates multiple overlapping rhythmic patterns that synchronize information across modules:

- Gamma oscillations (30-100 Hz) for local processing and working memory maintenance
- Theta oscillations (4-8 Hz) for episodic memory encoding and temporal sequencing
- Alpha oscillations (8-12 Hz) for attentional gating and inhibition of irrelevant information
- Delta oscillations (1-4 Hz) for global state regulation and sleep-phase consolidation

This approach, inspired by neural oscillations in the brain (Buzsáki & Draguhn, 2004), solves the binding problem in distributed representations and enables temporal coding impossible in traditional architectures.

### 6.2 Memory System Implementation

The NCA implements three distinct but interconnected memory systems:

#### 6.2.1 Working Memory

Working memory is maintained through persistent activity patterns in the prefrontal analog, stabilized by gamma oscillations. Information is encoded in a graph structure where nodes represent concepts and edges represent relationships. This allows for manipulation of complex relational structures like code syntax trees or experimental designs.

The working memory system implements active maintenance mechanisms with a limited capacity, forcing the system to prioritize information based on relevance to current goals.

#### 6.2.2 Episodic Memory

Episodic memory is encoded through hippocampal-inspired mechanisms where experiences are bound to temporal and contextual markers. The system uses "phase precession" where the timing of neural firing relative to theta oscillations encodes temporal sequences.

This approach enables the system to recall not just what happened but when and in what context, supporting temporal reasoning and learning from specific experiences.

#### 6.2.3 Semantic Memory

Semantic memory is consolidated during scheduled "sleep" phases where important connections are strengthened and irrelevant ones pruned. The system uses holographic encoding where information is distributed across the network, making it robust to partial damage and enabling content-addressable retrieval.

The consolidation process is guided by the glial regulatory network, which identifies important connections based on usage patterns and emotional tagging.

### 6.3 Specialized Module Development

#### 6.3.1 Software Engineering Module

The software engineering module implements specialized structures for code processing:

- Syntactic Processing Layer: Parses code into abstract syntax trees using recursive neural networks
- Semantic Analysis Layer: Builds meaning representations using graph neural networks
- Pattern Recognition Layer: Identifies common design patterns using memory-augmented networks
- Mental Model Formation: Integrates outputs into a comprehensive understanding of the codebase
- Counterfactual Reasoning: Simulates code execution under different conditions

This specialized architecture enables the system to understand code at multiple levels of abstraction, from syntax to architectural patterns.

#### 6.3.2 Scientific Reasoning Module

The scientific reasoning module implements a structured approach to hypothesis generation and testing:

- Observation Encoder: Transforms raw data into structured representations
- Hypothesis Generator: Uses a variational architecture to generate multiple possible explanations
- Prediction Engine: Derives testable predictions using causal inference mechanisms
- Experimental Design Optimizer: Generates experimental protocols that maximize information gain
- Evidence Evaluator: Updates belief distributions based on new evidence

This approach enables the system to engage in scientific reasoning that goes beyond pattern recognition to include causal inference and experimental design.

### 6.4 Metacognitive System

The metacognitive system maintains an active model of the NCA's own processing:

- State Monitoring: Tracks the current focus of attention, active goals, and resource allocation
- Uncertainty Representation: Explicitly models confidence levels for different knowledge domains
- Error Detection: Identifies inconsistencies in reasoning or conflicts between knowledge sources
- Resource Allocation: Dynamically adjusts computational resources based on task importance
- Explanation Generation: Traces the provenance of conclusions through the reasoning process

This self-referential processing loop enables the system to recognize its knowledge boundaries, express appropriate uncertainty, and explain its reasoning process transparently.

### 6.5 Global Workspace Implementation

The global workspace is implemented as a dynamic coalition of synchronized neural assemblies that temporarily dominate system-wide oscillatory patterns. Access to the workspace is controlled by a combination of bottom-up salience (regulated by the glial network) and top-down executive control (from the prefrontal analog).

This approach, inspired by Global Workspace Theory (Baars, 1997), enables information integration across specialized modules while reducing computational overhead compared to dense attention mechanisms.

### 6.6 Hardware Implementation

The NCA will be implemented through a hybrid hardware-software approach:

- Neural Processing: GPU clusters for parallel matrix operations
- Spiking Components: Neuromorphic chips (based on evolved versions of Intel's Loihi)
- Memory Hierarchy: Multi-tiered system with SRAM, phase-change memory, and high-density storage
- Temporal Coherence Engine: Custom hardware for generating and maintaining oscillatory patterns
- Interconnect: High-speed network fabric supporting multiple communication patterns

This hybrid approach leverages the strengths of different hardware platforms while maintaining implementability within 3-5 years using near-future technology.

---

## 7. Implementation Plan and Timeline

The implementation of the NeuroCognitive Architecture will follow a progressive, milestone-driven approach over 36 months:

### 7.1 Phase 1: Core Architecture Foundation (Months 1-6)

#### Milestone 1.1: Modular Framework Development (Months 1-2)
- Develop the base infrastructure for module communication and integration
- Implement the message-passing protocol between neural components
- Create the initial oscillatory binding mechanism prototype

#### Milestone 1.2: Memory Subsystem Prototype (Months 3-4)
- Implement the three-tier memory architecture (working, episodic, semantic)
- Develop basic memory encoding and retrieval mechanisms
- Create initial memory consolidation processes

#### Milestone 1.3: Proof-of-Concept Integration (Months 5-6)
- Integrate the modular framework with the memory subsystem
- Implement a simplified version of the triple-processing architecture
- Demonstrate basic information flow through the system

### 7.2 Phase 2: Specialized Module Development (Months 7-12)

#### Milestone 2.1: Prefrontal Analog & Executive Functions (Months 7-8)
- Implement working memory maintenance through gamma oscillations
- Develop attention direction mechanisms
- Create goal representation and maintenance systems

#### Milestone 2.2: Software Engineering Module (Months 9-10)
- Develop the recursive-hierarchical code processing structure
- Implement code comprehension and mental model formation capabilities
- Create pattern recognition systems for code analysis

#### Milestone 2.3: Scientific Reasoning Module (Months 11-12)
- Implement Bayesian prediction framework
- Develop hypothesis generation and testing mechanisms
- Create experimental design capabilities

### 7.3 Phase 3: Integration and Metacognition (Months 13-18)

#### Milestone 3.1: Global Workspace Implementation (Months 13-14)
- Develop the dynamic coalition formation mechanism
- Implement the sparse, graph-structured global workspace
- Create the access control system based on salience and executive control

#### Milestone 3.2: Metacognitive System (Months 15-16)
- Implement the self-referential processing loop
- Develop uncertainty recognition and representation
- Create the higher-order thought system for self-monitoring

#### Milestone 3.3: Emotional Tagging and Salience (Months 17-18)
- Implement the limbic analog for emotional processing
- Develop the salience detection system
- Create the novelty and utility evaluation mechanisms

### 7.4 Phase 4: Training and Optimization (Months 19-24)

#### Milestone 4.1: Multi-Phase Training Pipeline (Months 19-20)
- Develop the supervised learning component
- Implement reinforcement learning with intrinsic motivation
- Create the sleep-phase consolidation process

#### Milestone 4.2: Hardware Optimization (Months 21-22)
- Optimize neural components for GPU clusters
- Adapt spiking neural networks for neuromorphic hardware
- Implement the multi-tiered memory system across appropriate hardware

#### Milestone 4.3: System Integration and Validation (Months 23-24)
- Integrate all components into a cohesive system
- Perform comprehensive testing across domains
- Validate performance against baseline LLMs

### 7.5 Phase 5: Scaling and Specialization (Months 25-36)

#### Milestone 5.1: Scale-Up (Months 25-28)
- Increase model capacity across all components
- Optimize for distributed training and inference
- Implement dynamic resource allocation

#### Milestone 5.2: Domain Adaptation (Months 29-32)
- Specialize modules for targeted domains (software engineering, science)
- Fine-tune with domain-specific data
- Develop domain-specific interfaces

#### Milestone 5.3: Continuous Learning Implementation (Months 33-36)
- Develop mechanisms for ongoing learning from interactions
- Implement feedback incorporation systems
- Create long-term memory management for continuous operation
- Conduct final evaluation and documentation

### 7.6 Risk Mitigation Strategies

1. **Computational Complexity**: Begin with smaller-scale implementations of each module, then gradually scale up as integration proves successful.

2. **Integration Challenges**: Use frequent integration testing throughout development; implement clear interfaces between modules from the start.

3. **Hardware Limitations**: Develop software simulations of neuromorphic components that can later be replaced with specialized hardware as it becomes available.

4. **Training Data Requirements**: Start with synthetic data generation for specialized modules before moving to real-world data; implement curriculum learning approaches.

5. **Performance Validation**: Establish clear metrics for each module and the integrated system; develop benchmark tasks that specifically test brain-like capabilities.

---

## 8. Expected Results and Impact

### 8.1 Technical Outcomes

The successful implementation of the NeuroCognitive Architecture is expected to yield several significant technical outcomes:

1. **Improved Performance on Complex Reasoning Tasks**: The specialized modules for software engineering and scientific reasoning, combined with the integrated memory systems, should enable superior performance on tasks requiring domain expertise and contextual understanding.

2. **Enhanced Memory Capabilities**: The multi-layered memory system with active consolidation and pruning mechanisms should significantly improve the model's ability to maintain context over extended interactions and learn efficiently from experience.

3. **Metacognitive Awareness**: The self-referential processing loop should enable the system to recognize its knowledge boundaries, express appropriate uncertainty, and explain its reasoning process transparently.

4. **Computational Efficiency**: The brain-inspired architecture, with its selective attention mechanisms and specialized processing modules, should achieve greater computational efficiency than conventional transformer-based architectures.

5. **Continuous Learning Capabilities**: The sleep-phase consolidation process and glial-mediated pruning should enable the system to continuously learn and adapt without catastrophic forgetting.

### 8.2 Evaluation Metrics

The architecture's performance will be evaluated using several metrics:

1. **Task-Specific Performance**: Accuracy, precision, recall, and F1 scores on domain-specific tasks in software engineering and scientific reasoning.

2. **Memory Evaluation**: Performance on tasks requiring long-context understanding and episodic memory retrieval.

3. **Metacognitive Assessment**: Accuracy of uncertainty estimates, correlation between confidence and correctness, and quality of explanations.

4. **Efficiency Metrics**: Computational resource utilization, inference time, and energy consumption compared to baseline LLMs.

5. **Continuous Learning**: Performance improvement over time with ongoing interaction and feedback.

### 8.3 Broader Impact

The development of the NeuroCognitive Architecture has the potential for broad impact across multiple domains:

1. **Advancement of AI Research**: The architecture provides a new paradigm for AI development that more closely aligns with human cognitive processes, potentially opening new research directions in neuromorphic computing and cognitive architectures.

2. **Software Engineering**: The specialized software engineering module could significantly enhance developer productivity by providing more intelligent code comprehension, debugging assistance, and design suggestions.

3. **Scientific Discovery**: The scientific reasoning module could accelerate scientific discovery by generating novel hypotheses, designing experiments, and analyzing results across domains.

4. **Human-AI Collaboration**: The metacognitive capabilities should enable more effective human-AI collaboration by allowing the system to communicate its limitations and reasoning process transparently.

5. **Energy-Efficient AI**: The brain-inspired architecture, with its selective attention and specialized processing, could lead to more energy-efficient AI systems, reducing the environmental impact of large-scale AI deployment.

### 8.4 Novelty and Significance

The NeuroCognitive Architecture represents a significant advancement over existing approaches in several ways:

1. **Triple-Processing Architecture**: The combination of neural networks, glial regulatory systems, and oscillatory binding mechanisms in a unified architecture appears to be novel in AI research.

2. **Biologically-Inspired Memory Consolidation**: The implementation of distinct memory types with specialized encoding mechanisms and sleep-phase consolidation goes beyond existing memory-augmented neural networks.

3. **Oscillatory Binding Mechanism**: The use of multiple frequency bands for information synchronization and temporal encoding represents a novel approach to the binding problem in neural networks.

4. **Glial Regulatory Network**: The implementation of a separate regulatory system inspired by glial cells for modulation and resource allocation appears to be unique in AI architectures.

5. **Comprehensive Metacognition**: The implementation of metacognitive awareness through higher-order thought representations goes beyond existing uncertainty estimation approaches.

---

## 9. Conclusion

The proposed NeuroCognitive Architecture (NCA) represents a fundamental reimagining of large language models by implementing a multi-layered system that mirrors the human brain's structural and functional organization. By combining neural processing networks, glial regulatory systems, and oscillatory binding mechanisms in a unified architecture, the NCA addresses key limitations of current LLMs while improving computational efficiency.

The architecture's specialized modules for software engineering and scientific reasoning, combined with its multi-layered memory system and metacognitive capabilities, promise to advance AI capabilities in domains requiring expertise, contextual understanding, and self-awareness. The implementation plan provides a clear path forward, with progressive milestones and risk mitigation strategies to ensure successful development.

If successful, this research will not only advance the state of the art in AI but also deepen our understanding of how brain-inspired principles can be effectively translated into computational systems. The resulting architecture could serve as a foundation for a new generation of AI systems that more closely mirror human cognitive processes, leading to more capable, efficient, and trustworthy artificial intelligence.

---

## 10. References

Alayrac, J. B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., ... & Zisserman, A. (2022). Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35, 23716-23736.

Anderson, J. R., Bothell, D., Byrne, M. D., Douglass, S., Lebiere, C., & Qin, Y. (2004). An integrated theory of the mind. Psychological Review, 111(4), 1036-1060.

Araque, A., & Navarrete, M. (2010). Glial cells in neuronal network function. Philosophical Transactions of the Royal Society B: Biological Sciences, 365(1551), 2375-2381.

Baars, B. J. (1997). In the theater of consciousness: The workspace of the mind. Oxford University Press.

Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., ... & Sifre, L. (2022). Improving language models by retrieving from trillions of tokens. International Conference on Machine Learning, 2206-2240.

Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in Neural Information Processing Systems, 33, 1877-1901.

Bullmore, E., & Sporns, O. (2012). The economy of brain network organization. Nature Reviews Neuroscience, 13(5), 336-349.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. Science, 304(5679), 1926-1929.

Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., ... & Fiedel, N. (2022). PaLM: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. Behavioral and Brain Sciences, 36(3), 181-204.

Davies, M., Srinivasa, N., Lin, T. H., Chinya, G., Cao, Y., Choday, S. H., ... & Wang, H. (2018). Loihi: A neuromorphic manycore processor with on-chip learning. IEEE Micro, 38(1), 82-99.

Dehaene, S., Lau, H., & Kouider, S. (2017). What is consciousness, and could machines have it? Science, 358(6362), 486-492.

Friston, K. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.

Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. International Conference on Machine Learning, 1050-1059.

Graves, A., Wayne, G., Reynolds, M., Harley, T., Danihelka, I., Grabska-Barwińska, A., ... & Hassabis, D. (2016). Hybrid computing using a neural network with dynamic external memory. Nature, 538(7626), 471-476.

Hassabis, D., Kumaran, D., Summerfield, C., & Botvinick, M. (2017). Neuroscience-inspired artificial intelligence. Neuron, 95(2), 245-258.

Hawkins, J., & Blakeslee, S. (2004). On intelligence: How a new understanding of the brain will lead to the creation of truly intelligent machines. Macmillan.

Hemmer, P., Geller, E. H., Krebs, J. F., Keller, B., & Goel, A. K. (2023). AI systems that know when they don't know: A survey of uncertainty in machine learning. arXiv preprint arXiv:2302.11230.

Kendall, A., & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? Advances in Neural Information Processing Systems, 30.

Kumaran, D., Hassabis, D., & McClelland, J. L. (2016). What learning systems do intelligent agents need? Complementary learning systems theory updated. Trends in Cognitive Sciences, 20(7), 512-534.

Laird, J. E. (2012). The Soar cognitive architecture. MIT press.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33, 9459-9474.

Mahowald, K., Ivanova, A. A., Blank, I. A., Kanwisher, N., Tenenbaum, J. B., & Fedorenko, E. (2023). Dissociating language and thought in large language models: a cognitive perspective. arXiv preprint arXiv:2301.06627.

Manakul, P., Liusie, A., & Gales, M. J. (2023). Selfcheckgpt: Zero-resource black-box hallucination detection for generative large language models. arXiv preprint arXiv:2303.08896.

McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. Psychological Review, 102(3), 419-457.

Merolla, P. A., Arthur, J. V., Alvarez-Icaza, R., Cassidy, A. S., Sawada, J., Akopyan, F., ... & Modha, D. S. (2014). A million spiking-neuron integrated circuit with a scalable communication network and interface. Science, 345(6197), 668-673.

Reed, S., Zolna, K., Parisotto, E., Colmenarejo, S. G., Novikov, A., Barth-Maron, G., ... & de Freitas, N. (2022). A generalist agent. arXiv preprint arXiv:2205.06175.

Weston, J., Chopra, S., & Bordes, A. (2015). Memory networks. International Conference on Learning Representations.