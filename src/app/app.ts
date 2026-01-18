import { CommonModule } from '@angular/common';
import { Header } from './components/header/header';
import { About } from './components/about/about';
import { Experience } from './components/experience/experience';
import { Education } from './components/education/education';
import { Projects } from './components/projects/projects';
import { Certifications } from './components/certifications/certifications';
import { Skills } from './components/skills/skills';
import { Languages } from './components/languages/languages';
import { Component, OnInit, OnDestroy } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Footer } from './components/footer/footer';
@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    RouterOutlet,
    Header,
    About,
    Experience,
    Education,
    Projects,
    Certifications,
    Skills,
    Languages,
    Footer

  ],
  templateUrl: './app.html',
  styleUrls: ['./app.css']
})
export class App implements OnInit, OnDestroy {
  private _handler = (e: MouseEvent) => this.onMouseMove(e);

  ngOnInit(): void {
    if (typeof window !== 'undefined') {
    window.addEventListener('mousemove', this._handler) 
}
  }

  ngOnDestroy(): void {
    if (typeof window !== 'undefined') {
    window.removeEventListener('mousemove', this._handler);
  }}

  private onMouseMove(e: MouseEvent){
    const w = window.innerWidth;
    const h = window.innerHeight;
    const x = e.clientX / w; // 0..1
    const y = e.clientY / h; // 0..1
    const hue1 = Math.round(220 + x * 140); // 220..360
    const hue2 = Math.round(320 + y * 120); // 320..440 -> wrapped by HSL
    document.documentElement.style.setProperty('--hue1', String(hue1 % 360));
    document.documentElement.style.setProperty('--hue2', String(hue2 % 360));
  }
}