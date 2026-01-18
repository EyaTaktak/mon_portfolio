import { Component } from '@angular/core';
import { CERTIFICATIONS } from '../../data';
import { CommonModule } from '@angular/common';
@Component({
  selector: 'app-certifications',
  standalone: true,  
  imports: [CommonModule],
  templateUrl: './certifications.html',
  styleUrls: ['./certifications.css'],
})
export class Certifications {
  list = CERTIFICATIONS;
}
