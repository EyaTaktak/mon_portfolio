# Étape 1 : Build
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . ./
RUN npx ng build --configuration production

# Étape 2 : Serveur Nginx
FROM nginx:stable-alpine
# Note le chemin : dist/portfolio/browser
COPY --from=build /app/dist/portfolio/browser /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]