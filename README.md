# Portfolio — English

This repository contains my personal portfolio built with Angular. The site displays About, Projects, Certifications, Skills, and Experience sections with a dynamic color theme that responds to mouse movement.

Quick start (development)

1. Install dependencies:

```bash
npm install
```

2. Run dev server:

```bash
ng serve
```

Open http://localhost:4200

Build (production)

```bash
ng build --configuration production
```

Docker (build and serve)

1. Build the Docker image:

```bash
docker build -t portfolio-app .
```

2. Run the container:

```bash
docker run -p 8080:80 portfolio-app
```

Open http://localhost:8080

