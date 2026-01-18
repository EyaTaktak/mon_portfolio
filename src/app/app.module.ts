import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { RouterModule } from '@angular/router';

import { App } from './app'; // Ton app.ts

// Importer tous tes composants
import { Header} from './components/header/header';
import { About} from './components/about/about';
import { Experience } from './components/experience/experience';
import { Education } from './components/education/education';
import { Projects} from './components/projects/projects';
import { Certifications } from './components/certifications/certifications';
import { Skills} from './components/skills/skills';
import { Languages } from './components/languages/languages';

@NgModule({
  imports: [
    BrowserModule,
    RouterModule.forRoot([]),
    App,
    Header,
    About,
    Experience,
    Education,
    Projects,
    Certifications,
    Skills,
    Languages
  ],
  bootstrap: [App]
})
export class AppModule { }
