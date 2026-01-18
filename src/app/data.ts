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
    school: "IPEIT __ Preparatory Cycle",
    schoolUrl: "http://www.ipeit.rnu.tn/",
    degree: "Mathematics and Physics Student (MP)",
    period: "September 2021 – May 2023",
    courses: ["Analysis", "Algebra", "Python", "SQL"]
  },
  {
    school: "Kheireddine High School, Ariana",
    schoolUrl: "",
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
    name: 'Energy Optimization for Buildings',
    context: 'Green AI Hackathon',
    description: `
Unsupervised learning system to detect energy waste in office buildings
and simulate optimized energy consumption.
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
      'Energy Optimization Simulation',
      'Visualization and Reporting'
    ],
    link: 'https://www.kaggle.com/code/eyats1/optim-energy'
  },
  {
    name: 'Solar Panel Image Classification',
    context: 'Green AI Hackathon',
    description: `
CNN-based model using ResNet18 to classify solar panel conditions
for predictive maintenance.
    `,
    date: '2025',
    tools: [
      'PyTorch',
      'Torchvision',
      'CNN',
      'Transfer Learning',
      'Data Augmentation'
    ],
    skills: [
      'Convolutional Neural Networks (CNN)',
      'Image Classification',
      'Data Preprocessing & Augmentation',
      'Transfer Learning',
      'Model Evaluation and Optimization',
      'Predictive Maintenance Applications'
    ],
    link: 'https://www.kaggle.com/code/eyats1/classification-des-images-de-pv-d-fectueux'
  },
  {
    name: 'Air Quality Classification (Multimodal AI)',
    context: 'ENSTA-B Project',
    description: `
Multimodal deep learning system combining images and sensor data.
Databricks ETL pipeline using Medallion Architecture.
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
      'Multimodal Data Integration',
      'Deep Learning Modeling',
      'ETL Pipelines',
      'Data Cleaning & Transformation',
      'Feature Engineering',
      'Air Quality Prediction',
      'Model Deployment & Evaluation'
    ],
    link: 'https://www.kaggle.com/code/eyats1/multi-modal-air-quality-classification'
  },
  {
    name: 'Medical Office Platform',
    context: 'ENSTA-B Project',
    description: `
Intelligent web platform for medical appointment management
with a deep learning model for skin disease prediction.
    `,
    date: '2024',
    tools: [
      'Flask',
      'MySQL',
      'HTML',
      'CSS',
      'Deep Learning',
      'Transfer Learning'
    ],
    skills: [
      'Full-Stack Web Development',
      'Database Design & Management',
      'Backend API Development',
      'Deep Learning for Image Analysis',
      'Transfer Learning',
      'UX/UI Design',
      'Healthcare Application Development'
    ],
    link: 'https://github.com/EyaTaktak/Cabinet_Medical'
  },
  {
    name: 'Intelligent Medical Drone Delivery Platform',
    context: 'ENSTA-B Project',
    description: `
Autonomous medical delivery system with object detection
and a mobile application for order management.
    `,
    date: '2025',
    tools: [
      'YOLO',
      'OpenCV',
      'Python',
      'Flutter',
      'Dart',
      'Firebase'
    ],
    skills: [
      'Object Detection & Tracking',
      'Computer Vision for Autonomous Navigation',
      'Mobile Application Development',
      'Backend Integration with Firebase',
      'Autonomous Systems',
      'User Interface & Experience Design'
    ],
    link: 'https://github.com/EyaTaktak/Drone-Application'
  }
];


// ===========================
// CERTIFICATIONS
// ===========================
export const CERTIFICATIONS = [
  {
    name: 'Building LLM Applications With Prompt Engineering',
    issuer: 'NVIDIA',
    date: '2025',
    link: 'https://learn.nvidia.com/certificates?id=b2KTSeDCTvarUQ3AQgsk1A'
  },
  {
    name: 'Advanced Artificial Intelligence Training: From Machine Learning to Deep Learning Deployment',
    issuer: 'ENSTA-B',
    date: '2025',
    link: 'https://drive.google.com/file/d/1vs4InVJs0UL0PYyHLNbCtZHqljJ3W3uL/view?usp=drive_link'
  },
  {
    name: 'AWS Educate Machine Learning Foundations - Training Badge',
    issuer: 'AWS',
    date: '2025',
    link: 'https://www.credly.com/badges/8e278b16-e13f-4d98-bbfa-797f109740f4'
  },
  {
    name: 'Get Started with Databricks for Data Engineering',
    issuer: 'Databricks',
    date: '2025',
    link: 'https://drive.google.com/file/d/1Ff2J9hoXY1v8Hfn6WUzemmemkN_1rKOZ/view?usp=sharing'
  },
  {
    name: 'Introduction to Deep Learning and Neural Networks with Keras',
    issuer: 'IBM',
    date: '2024',
    link: 'https://www.coursera.org/account/accomplishments/verify/RNB6DAKA34LV'
  },
  {
    name: 'Deep Learning with TensorFlow 2',
    issuer: '365 Data Science',
    date: '2024',
    link: 'https://learn.365datascience.com/certificates/CC-AA10C0880D/'
  }
];

// ===========================
// SKILLS
// ===========================
export const SKILLS = [
  {
    category: "Languages & Web",
    skills: ['Python', 'C/C++', 'Dart', 'Java', 'JavaScript', 'SQL', 'HTML', 'CSS', 'Angular'],
    color: "blue"
  },
  {
    category: "AI & Data Science",
    skills: [
      'Machine Learning', 'Deep Learning', 'Computer Vision', 'NLP', 
      'Transfer Learning', 'Fine-Tuning', 'LLM', 'Prompt Engineering', 
      'RAG', 'LangChain'
    ],
    color: "cyan"
  },
  {
    category: "Frameworks & Tools",
    skills: ['PyTorch', 'TensorFlow / Keras', 'Scikit-learn', 'NumPy', 'Pandas', 'OpenCV', 'YOLO', 'Flask'],
    color: "purple"
  },
  {
    category: "Cloud, DB & Soft Skills",
    skills: ['MySQL', 'Firebase', 'Databricks', 'Problem Solving', 'Teamwork', 'Adaptability', 'Communication'],
    color: "green"
  }
];