import { Component } from '@angular/core';
import { EXPERIENCE } from '../../data';
import { CommonModule } from '@angular/common';
@Component({
  selector: 'app-experience',
  standalone: true,  
  imports: [CommonModule],
  templateUrl: './experience.html',
  styleUrl: './experience.css',
})
export class Experience {
  list: any[]= EXPERIENCE;
  
}
