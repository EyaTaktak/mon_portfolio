import { Component } from '@angular/core';
import { ABOUT } from '../../data';
import { CommonModule } from '@angular/common';
@Component({
  selector: 'app-about',
  standalone: true,  
  imports: [CommonModule],
  templateUrl: './about.html',
  styleUrl: './about.css',
})
export class About {
  info = ABOUT;
  myProfilePic = 'assets/Eya_Taktak.jpg';
  // Cette fonction supprime l'erreur TS2339
  handleImageError(event: any) {
    // Si l'image locale ne charge pas, on utilise une image de secours
    event.target.src = 'https://ui-avatars.com/api/?name=Eya+Taktak&background=00d1ff&color=fff&size=280';
    console.error("L'image n'a pas pu être chargée au chemin indiqué.");
}}
