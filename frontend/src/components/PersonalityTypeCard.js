// src/components/PersonalityTypeCard.js
import React from 'react';
import { Card, CardContent, Typography, Box, Chip } from '@mui/material';

const personalityColors = {
  'Extrovertido': '#FF6384',
  'Neurótico': '#36A2EB',
  'Amable': '#FFCE56',
  'Concienzudo': '#4BC0C0',
  'Abierto': '#9966FF',
  'Reservado': '#FF9F40',
  'Emocionalmente estable': '#8AC24A',
  'Antipático': '#F06292',
  'Poco concienzudo': '#7986CB',
  'Cerrado a experiencias': '#A1887F'
};

const PersonalityTypeCard = ({ cluster, type }) => {
  return (
    <Card sx={{ mb: 2 }}>
      <CardContent>
        <Box display="flex" alignItems="center" mb={1}>
          <Typography variant="h6" component="div">
            Cluster {cluster}:
          </Typography>
          <Chip 
            label={type} 
            sx={{ 
              ml: 1,
              backgroundColor: personalityColors[type.split(' ')[0]] || '#e0e0e0',
              color: 'white',
              fontWeight: 'bold'
            }} 
          />
        </Box>
        <Typography variant="body2" color="text.secondary">
          {getPersonalityDescription(type)}
        </Typography>
      </CardContent>
    </Card>
  );
};

const getPersonalityDescription = (type) => {
  const descriptions = {
    'Extrovertido': 'Personas sociables, habladoras y que disfrutan de la compañía de otros.',
    'Neurótico': 'Personas que experimentan emociones negativas con mayor facilidad.',
    'Amable': 'Personas cooperativas, compasivas y que buscan la armonía.',
    'Concienzudo': 'Personas organizadas, responsables y con autocontrol.',
    'Abierto': 'Personas creativas, curiosas y con intereses diversos.',
    'Reservado': 'Personas más tranquilas, reservadas y que prefieren la soledad.',
    'Emocionalmente estable': 'Personas calmadas, relajadas y con menor reactividad emocional.',
    'Antipático': 'Personas más escépticas y menos cooperativas.',
    'Poco concienzudo': 'Personas más espontáneas y menos organizadas.',
    'Cerrado a experiencias': 'Personas más tradicionales y con intereses más convencionales.'
  };
  
  return descriptions[type.split(' ')[0]] || 'Descripción no disponible.';
};

export default PersonalityTypeCard;