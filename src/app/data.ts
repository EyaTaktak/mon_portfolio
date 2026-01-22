// ===========================
// ABOUT
// ===========================
export const ABOUT = {
  name: 'Eya Taktak',
  title: 'Engineering Student in Advanced Technologies',
  location: 'La Soukra, Ariana, Tunisia',
  email: 'eya.taktak@enstab.ucar.tn',
  phone: '+216 96 777 116',
  linkedin: 'https://www.linkedin.com/in/eya-taktak/',
  github: 'https://github.com/EyaTaktak',
  summary: `
Third-year Advanced Technologies Engineering student at ENSTA-B, passionate about Artificial Intelligence
and intelligent systems. I have hands-on experience in deep learning, computer vision, and full-stack
development through academic projects, hackathons, and professional internships.
Currently seeking a Final Year Project (PFE) starting February 2026.
  `
};

// ===========================
// EXPERIENCE
// ===========================
export const EXPERIENCE = [
  {
    role: 'Technical Internship – Check Amount Prediction Project',
    company: 'BFI Group',
    logo: 'assets/bfi_logo.svg',
    link: 'https://bfigroupe.com/',
    duration: 'August – September 2025',
description: `Designed and implemented a complete end-to-end check image processing pipeline, from image preprocessing to handwritten text recognition. 
Performed advanced dataset cleaning, including handling missing values, removing duplicates, and eliminating noisy or inconsistent samples to improve data quality. 
Merged multiple handwritten datasets to increase training data volume and enhance handwriting style diversity, improving model generalization.

Conducted a comparative study of multiple deep learning models to identify the most suitable architecture for handwritten amount recognition.

Trained and evaluated the selected architecture, achieving approximately 95% accuracy on validation data.
Optimized the training process using early stopping and hyperparameter tuning to prevent overfitting and improve convergence.
Developed and deployed a desktop application using Tkinter to visualize check images and test real-time predictions.`,   
skills: [
  'Deep Learning',
  'Computer Vision',
  'Handwritten Text Recognition (HTR)',
  'CRNN Models','LSTM',
  'Image Preprocessing & Augmentation',
  'Dataset Cleaning & Preparation',
  'Handling Missing Values & Duplicates',
  'Dataset Fusion & Data Engineering',
  'Model Evaluation & Benchmarking',
  'Early Stopping & Overfitting Prevention',
  'Model Optimization & Hyperparameter Tuning',
  'Python',
  'TensorFlow / Keras',
  'Tkinter (Desktop Application Development)',
  'AI Model Deployment'
]

  },
  {
    role: 'Worker Internship – AI Exploration',
    company: 'BFI Group',
    link: 'https://bfigroupe.com/',
    logo: 'assets/bfi_logo.svg',
    duration: 'July 2025',
    description: `Explored and fine-tuned multiple artificial intelligence and deep learning models for document analysis and computer vision tasks.
Conducted comparative experiments to evaluate model performance, robustness, and suitability for real-world check processing use cases.
Designed and implemented an automatic check zone detection module using object detection techniques to accurately localize key regions within check images.
Trained and evaluated an object detection model to identify critical check areas, contributing to improved preprocessing and downstream recognition accuracy.
Analyzed detection results and iteratively improved model performance through data preparation and parameter tuning.`,
    


skills: [
      'Artificial Intelligence',
    'Deep Learning',
    'Computer Vision',
    'Object Detection',
    'Model Fine-Tuning',
    'Model Evaluation & Comparison',
    'Faster R-CNN',
    'Dataset Preparation',
    'Image Annotation',
    'Python',
    'TensorFlow / PyTorch'
    ]
  }
];

// ===========================
// EDUCATION
// ===========================
export const EDUCATION = [
  {
    school: "ENSTA-B | Engineering Cycle",
    schoolUrl: "https://enstab.tn/",
    degree: "Advanced Technologies Student (AT)",
    period: "September 2023 – Present",
    logo: 'assets/Enstab-logo.png',
    courses: [
      "Cloud Computing",
      "Java",
      "Artificial Intelligence",
      "Optimization",
      "Numerical Analysis",
      "Image Processing"
    ]
  },
  {
    school: "IPEIT | Preparatory Cycle",
    schoolUrl: "http://www.ipeit.rnu.tn/",
    logo: 'assets/ipeit-logo.jpg',
    degree: "Mathematics and Physics Student (MP)",
    period: "September 2021 – May 2023",
    courses: ["Analysis", "Algebra", "Python", "SQL"]
  },
  {
    school: "Kheireddine High School, Ariana",
    schoolUrl: "",
    logo: 'assets/lycee.png',
    degree: "Experimental Sciences Baccalaureate",
    period: "September 2020 – May 2021",
    courses: []
  }
];
// ===========================
// PROJECTS
// ===========================
export const PROJECTS = [ 
  {
    name: 'Personal Portfolio Website',
    context: 'Personal Project',
    description: `
Architected a production-ready infrastructure using multi-stage Docker builds and an optimized Nginx configuration for
high-performance SPA routing and scalability. Deployed the Angular web application on Render with CI/CD pipelines via GitHub Actions for automated testing and deployment.`,
    date: '2026',
    tools: [
      'Angular',
      'TypeScript',
      'HTML',
      'CSS',
      "CI/CD",
      "Docker",
      "nginx","Github Actions",
      "Render"
    ],
    skills: [
      'Web Development',"Devops"
    ],
    link: 'https://eya-portfolio.onrender.com/'
  },
  {
    name: 'Customer Sentiment Analysis with LangChain',
    context: 'NVIDIA LLM Applications Course',
    description: `
LangChain pipeline analyzing customer emails to identify product categories with most complaints and store locations
with highest negative sentiment.`,
    tools: [
      "Python",
      "LangChain",
      "NLP",
      "Sentiment Analysis",
      "Llama 3.1",
      "Pandas"
    ],
    skills: [
      'LLM Applications',
      'Sentiment Analysis',
      'Data Analysis'
    
    ],
    link: 'https://sentiment-analyser-f90e.onrender.com/'
  },
{
    name: 'LLM Applications Development with LangChain',
    date: '2025',
    context: 'NVIDIA LLM Applications Course',
    description: `
– Designed and implemented end-to-end LLM pipelines using LangChain, including prompt engineering, structured outputs,
batch and streaming inference, conversation memory, and agent-based reasoning with tool integration for text analysis and
automation tasks.
`,
    tools: ['Python', 'LangChain', 'OpenAI API', 'Prompt Engineering', 'LCEL', 'NLP', 'Pandas'],
    skills: [
      'LLM Applications',
      'Prompt Engineering',
      'Structured Outputs',
      'Streaming Inference',
      'Conversation Memory',
      'Agent-Based Reasoning'
    ]
  },
  {
    name: 'Energy Optimization for Buildings',
    context: 'Green AI Hackathon',
    description: `
Unsupervised learning system to detect energy waste
in office buildings and simulate optimized energy consumption.
    `,
    date: '2025',
    tools: [
      'Python',
      'Pandas',
      'NumPy',
      'Scikit-learn',
      'Matplotlib',
      'Isolation Forest'
    ],
    skills: [
      'Data Analysis',
      'Feature Engineering',
      'Unsupervised Learning',
      'Anomaly Detection',
      'Energy Optimization',
      'Visualization'
    ],
    link: 'https://www.kaggle.com/code/eyats1/optim-energy'
  },
  {
    name: 'Solar Panel Image Classification',
    context: 'Green AI Hackathon',
    description: `
ResNet18-based CNN model to classify the condition of solar panels
for predictive maintenance. Achieved ~95% accuracy.
    `,
    date: '2025',
    tools: [
      'PyTorch',
      'Torchvision',
      'CNN',
      'ResNet18',
      'Transfer Learning',
      'Data Augmentation'
    ],
    skills: [
      'CNN',
      'Image Classification',
      'Data Preprocessing',
      'Data Augmentation',
      'Transfer Learning',
      'Model Evaluation'
    ],
    link: 'https://www.kaggle.com/code/eyats1/classification-des-images-de-pv-d-fectueux'
  },
  {
    name: 'Air Quality Classification (Multimodal AI)',
    context: 'ENSTA-B Project',
    description: `
Multimodal deep learning system combining images and sensor data
with an ETL pipeline in Databricks (Medallion Architecture).
    `,
    date: '2025',
    tools: [
      'Python',
      'PyTorch',
      'Apache Spark',
      'Databricks',
      'Multimodal Learning',
      'Computer Vision'
    ],
    skills: [
      'Feature Engineering',
      'Multimodal Learning',
      'Deep Learning',
      'ETL Pipelines',
      'Data Cleaning',
      'Air Quality Prediction',
      'Model Deployment'
    ],
    link: 'https://www.kaggle.com/code/eyats1/multi-modal-air-quality-classification'
  },
  {
    name: 'Medical Office Platform',
    context: 'ENSTA-B Project',
    description: `
Intelligent web platform for managing medical appointments
with a VGG16 CNN model to predict skin diseases (accuracy ~97%).
    `,
    date: '2025',
    tools: [
      'Flask',
      'MySQL',
      'HTML',
      'CSS',
      'Xampp',
      'Deep Learning',
      'Transfer Learning',
      'VGG16'
    ],
    skills: [
      'Full-Stack Development',
      'Database Design',
      'API Development',
      'CNN',
      'VGG16',
      'Transfer Learning',
      'UX/UI Design'
    ],
    link: 'https://github.com/EyaTaktak/Cabinet_Medical'
  },{
    name: 'Object Detection for Medical Drone Delivery Platform',
    context: 'ENSTA-B Project',
    description: `
Object detection system for autonomous medical drones
using YOLO and OpenCV for navigation.
    `,
    date: '2024',
    tools: [
      'YOLO',
      'OpenCV',
      'Python',
      'Computer Vision'
    ],
    skills: [
      'Object Detection',
      'Computer Vision',
      'Autonomous Navigation'
    ],
    link: 'https://github.com/EyaTaktak/Yolo_model'
  },
  {
    name: 'Intelligent Medical Drone Delivery Mobile Application',
    context: 'ENSTA-B Project',
    description: `
Mobile application to manage orders and deliveries using autonomous drones.
    `,
    date: '2024',
    tools: [
      'Flutter',
      'Dart',
      'Firebase'
    ],
    skills: [
      'Mobile Development',
      'Firebase Integration',
      'UI/UX Design'
    ],
    link: 'https://github.com/EyaTaktak/Drone-Application'
  },
  
];


// ===========================
// CERTIFICATIONS
// ===========================
export const CERTIFICATIONS = [
  {
    name: 'Building LLM Applications With Prompt Engineering',
    issuer: 'NVIDIA',
    date: 'dec 2025',
    logo: 'assets/nvidia_logo.png',
    link: 'https://learn.nvidia.com/certificates?id=b2KTSeDCTvarUQ3AQgsk1A'
  },
  {
    name: 'AWS Educate Machine Learning Foundations - Training Badge',
    issuer: 'AWS',
    date: 'dec 2025',
    logo: 'assets/aws.svg',
    link: 'https://www.credly.com/badges/8e278b16-e13f-4d98-bbfa-797f109740f4'
  },
  {
    name: 'Advanced Artificial Intelligence Training: From Machine Learning to Deep Learning Deployment',
    issuer: 'ENSTA-B',
    date: 'oct 2025 - dec 2025',
    logo: 'assets/Enstab-logo.png',
    link: 'https://drive.google.com/file/d/1vs4InVJs0UL0PYyHLNbCtZHqljJ3W3uL/view?usp=drive_link'
  },
  
  {
    name: 'Get Started with Databricks for Data Engineering',
    issuer: 'Databricks',
    date: 'dec 2025',
    logo: 'assets/databricks_logo.png',
    link: 'https://drive.google.com/file/d/1Ff2J9hoXY1v8Hfn6WUzemmemkN_1rKOZ/view?usp=sharing'
  },{
    name: 'Foundations of Web Development: CSS, Bootstrap, JS, React',
    issuer: 'Udemy',
    date: 'Apr 2025',
    logo: 'assets/udemy-logo.png',
    link: 'https://www.udemy.com/certificate/UC-a5ed768a-f25d-4f73-800b-914966a752d6/'
  },

  {
    name: 'Hashgraph Developper Course',
    issuer: 'Hedera',
    date: 'Jan 2025',
    logo: 'assets/hedera.png',
    link: 'https://www.google.com/url?sa=D&q=https://certs.hashgraphdev.com/b523b6cc-b5a3-4a0e-9e04-846cb48df53b.pdf&ust=1768855740000000&usg=AOvVaw1jle-27xg0L3uQM3z72oqY&hl=fr&source=gmail'
  },
  {
    name: 'Deep Learning with TensorFlow 2',
    issuer: '365 Data Science',
    date: 'Nov 2024',
    logo: 'assets/365_data_science_logo.png',
    link: 'https://learn.365datascience.com/certificates/CC-AA10C0880D/'
  },{
    name: 'Git and GitHub',
    issuer: '365 Data Science',
    date: 'Nov 2024',
    logo: 'assets/365_data_science_logo.png',
    link: 'https://learn.365datascience.com/certificates/CC-3899D6734A/'
  },

  {
    name: 'Introduction to Deep Learning and Neural Networks with Keras',
    issuer: 'IBM',
    date: 'Jun 2024',
    logo: 'assets/ibm_logo.svg',
    link: 'https://www.coursera.org/account/accomplishments/verify/RNB6DAKA34LV'
  }
  
  
];

// ===========================
// SKILLS
// ===========================
export const SKILLS = [
  
  {
    category: "Artificial Intelligence & Machine Learning",
    skills: [
      'Machine Learning',
      'Deep Learning',
      'Computer Vision',
      'Natural Language Processing (NLP)',
      'Transfer Learning',
      'Model Fine-Tuning',
      'Large Language Models (LLMs)',
      'Prompt Engineering',
      'Retrieval-Augmented Generation (RAG)',
      'LangChain',
      'Model Evaluation & Optimization'
    ],
    icone: 'assets/assistant-ia1.gif',
    color: "blue"
  },
  {
    category: "Data Engineering & ETL",
    skills: [
      'ETL / ELT Pipelines',
      'Data Cleaning & Validation',
      'Data Transformation',
      'Medallion Architecture (Bronze / Silver / Gold)',
      'Batch Processing',
      'Data Quality & Monitoring'
    ],
    icone: 'assets/prg.gif',
    color: "cyan"
  },
  {
    category: "Frameworks, Libraries & Visualization",
    skills: [
      'PyTorch',
      'TensorFlow / Keras',
      'Scikit-learn',
      'NumPy',
      'Pandas',
      'OpenCV',
      'YOLO',
      'Matplotlib',
      'Seaborn',
      'Power BI'
    ],
    icone: 'assets/librairie1.gif',
    color: "teal"
  },
  {
    category: "Programming Languages & Web Technologies",
    skills: [
      'Python',
      'C / C++',
      'Java',
      'JavaScript',
      'TypeScript',
      'Dart',
      'SQL',
      'HTML5',
      'CSS3',
      'Angular',
      'Flask'
    ],
    icone: 'assets/1.gif',
    color: "purple"
  },
  {
    category: "Cloud, DevOps & Data Platforms",
    skills: [
      'Databricks',
      'Apache Spark (PySpark)',
      'Docker',
      'CI / CD Pipelines',
      'Firebase',
      'Linux Basics',
      'Git & GitHub'
    ],
    icone: 'assets/cloud.gif',
    color: "green"
  },
  {
    category: "Databases & Analytics Tools",
    skills: [
      'MySQL',
      'SQL Optimization',
      'Data Modeling',
      'Excel'
    ],
    icone: 'assets/db.png',
    color: "yellow"
  },
  {
    category: "Soft Skills",
    skills: [
      'Problem Solving',
      'Analytical Thinking',
      'Critical Thinking',
      'Teamwork & Collaboration',
      'Communication',
      'Adaptability',
      'Autonomous Learning'
    ],
    icone: 'assets/soft_skills.svg',
    color: "orange"
  }
];

