import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { SKILLS } from '../../data'; // Vérifie bien que le chemin vers ton fichier data.ts est correct

@Component({
  selector: 'app-skills',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './skills.html',
  styleUrl: './skills.css'
})
export class Skills {
  // Cette ligne est indispensable pour supprimer l'erreur TS2339
  list = SKILLS; 
}