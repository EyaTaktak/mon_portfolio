# Portfolio — English

This repository contains a personal portfolio built with Angular. The site displays About, Projects, Certifications, Skills, and Experience sections with a dynamic color theme that responds to mouse movement.

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

What I added or improved

- Multi-stage `Dockerfile` and `nginx.conf` for serving the built app.
- Global dynamic color background (mouse-driven) in `src/styles.css`.
- `src/app/data.ts` contains structured sample data for Projects, Certifications, Skills, Experience and About.
- Components updated to read from `data.ts` and render content in English.

If you want more custom content (your own projects, links, or translations), open `src/app/data.ts` and replace the sample entries.

If you'd like, I can also:

- Add more styling or animations
- Export a printable résumé page
- Add contact form with form handling

