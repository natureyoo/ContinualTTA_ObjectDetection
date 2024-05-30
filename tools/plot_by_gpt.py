import matplotlib.pyplot as plt

# Data
fixed_tau1 = {
    'SwinT-Evenly-Skip': {'FPS': [10.77731375, 12.89698533, 15.20334474, 18.05],
                          'mAP': [22.2, 21.9, 21.2, 20.6],
                          'labels': ['1 per 2', '1 per 3', '1 per 5', '1 per 10']},
    'SwinT-Ours-Skip': {'FPS': [17.03577513, 17.05, 17.27, 17.35, 17.45, 17.58],
                   'mAP': [21.72240454, 21.7, 21.64097137, 21.59528936, 21.58, 21.57],
                   'labels': ['1.0', '1.05', '1.1', '1.15', '1.2', 'inf']},
    'R50-Evenly-Skip': {'FPS': [11.7, 14.23, 17.03940362, 20.79],
                        'mAP': [21.9, 21.7, 21.2, 20.36],
                        'labels': ['1 per 2', '1 per 3', '1 per 5', '1 per 10']},
    'R50-Ours-Skip': {'FPS': [17.13, 20.3757764, 21.24, 21.32334531, 21.33553635, 21.38735984, 21.53075926],
                 'mAP': [22.13, 21.99322769, 21.9, 21.72173945, 21.65, 21.6, 21.1],
                   'labels': ['1.0', '1.05', '1.1', '1.15', '1.2', 'inf']},
}
fixed_tau2 = {
    'SwinT-Evenly-Skip': {'FPS': [10.77731375, 12.89698533, 15.20334474, 18.05],
                          'mAP': [22.2, 21.9, 21.2, 20.6],
                          'labels': ['1 per 2', '1 per 3', '1 per 5', '1 per 10']},
    'SwinT-Ours-Skip': {'FPS': [17.0575693, 18.00680272, 18.74853527, 19.13875598],
                   'mAP': [21.72262772, 21.47892495, 20.7605992, 20.61],
                   'labels': ['1.0', '1.1', '1.2', '1.5', 'inf']},
    'R50-Evenly-Skip': {'FPS': [11.7, 14.23, 17.03940362, 20.79],
                        'mAP': [21.9, 21.7, 21.2, 20.36],
                         'labels': ['1 per 2', '1 per 3', '1 per 5', '1 per 10']},
    'R50-Ours-Skip': {'FPS': [20.9, 21.24150894, 21.51944298, 22.01115891, 23.0606317],
                 'mAP': [22.02, 21.85940011, 21.58647527, 21.05755933, 19.83],
                   'labels': ['1.0', '1.1', '1.2', '1.5', 'inf']},
}

# Set the style to Apple Keynote style
plt.style.use('default')  # Reset to default style first
plt.style.use({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 20
})


# Create a single scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
# fig.suptitle('Scatter Plot of FPS vs mAP (Keynote Style)')

# Plot each scatter plot with consistent colors and markers
for (name, values) in fixed_tau1.items():
    if 'SwinT' in name:
        # color = '#6F777D'  # Light gray-blue for SwinT
        color = '#0071E3'
    else:
        color = '#E87530'  # Burnt orange for R50
    
    if 'Evenly-Skip' in name:
        marker = 'x'
    else:
        marker = 's'
    
    # ax.scatter(values['FPS'], values['mAP'], label=name, color=color, marker=marker, alpha=alpha, edgecolors='none')
    ax.scatter(values['FPS'], values['mAP'], label=name, color=color, marker=marker, edgecolors='none')
    # Adding labels
    if 'Evenly-Skip' in name:
        offset = 0.1
        for i, label in enumerate(values['labels']):
            ax.text(values['FPS'][i] + offset, values['mAP'][i] + offset, label, fontsize=15, ha='right', va='bottom')
    else:
        offset = 0.1
        for i, label in enumerate(values['labels']):
        	if label in ['1.0', '1.1', '1.2']:
        		ax.text(values['FPS'][i] + 5*offset, values['mAP'][i], label, fontsize=15, ha='right', va='bottom')
        	elif label == 'inf':
        		ax.text(values['FPS'][i] + 5*offset, values['mAP'][i]-offset, label, fontsize=15, ha='right', va='bottom')


# Adding lines
ax.plot(fixed_tau1['SwinT-Evenly-Skip']['FPS'], fixed_tau1['SwinT-Evenly-Skip']['mAP'], linestyle='-', color='#0071E3', alpha=1.0, linewidth=2.5)
ax.plot(fixed_tau1['R50-Evenly-Skip']['FPS'], fixed_tau1['R50-Evenly-Skip']['mAP'], linestyle='-', color='#E87530', alpha=1.0, linewidth=2.5)
ax.plot(fixed_tau1['SwinT-Ours-Skip']['FPS'], fixed_tau1['SwinT-Ours-Skip']['mAP'], linestyle='-', color='#0071E3', alpha=1.0, linewidth=2.5)
ax.plot(fixed_tau1['R50-Ours-Skip']['FPS'], fixed_tau1['R50-Ours-Skip']['mAP'], linestyle='-', color='#E87530', alpha=1.0, linewidth=2.5)

# Adding labels and title
ax.set_xlabel('FPS', fontname='Times New Roman', fontsize=25)
ax.set_ylabel('mAP', fontname='Times New Roman', fontsize=25)

# Setting y-axis range
ax.set_ylim(19.5, 23)

# Adding a legend
ax.legend(fontsize=15)

# Save the figure as PDF and PNG
plt.savefig('fig_FixedTau1.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig_FixedTau1.png', format='png', bbox_inches='tight')


# Create a single scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
# fig.suptitle('Scatter Plot of FPS vs mAP (Keynote Style)')

# Plot each scatter plot with consistent colors and markers
for (name, values) in fixed_tau2.items():
    if 'SwinT' in name:
        # color = '#6F777D'  # Light gray-blue for SwinT
        color = '#0071E3'
    else:
        color = '#E87530'  # Burnt orange for R50
    
    if 'Evenly-Skip' in name:
        marker = 'x'
    else:
        marker = 's'
    
    # ax.scatter(values['FPS'], values['mAP'], label=name, color=color, marker=marker, alpha=alpha, edgecolors='none')
    ax.scatter(values['FPS'], values['mAP'], label=name, color=color, marker=marker, edgecolors='none')
    # Adding labels
    if 'Evenly-Skip' in name:
        offset = 0.1
        for i, label in enumerate(values['labels']):
            ax.text(values['FPS'][i] + offset, values['mAP'][i] + offset, label, fontsize=15, ha='right', va='bottom')
    else:
        offset = 0.1
        for i, label in enumerate(values['labels']):
        	if label in ['1.0', '1.1', '1.2', '1.5']:
        		ax.text(values['FPS'][i] + 5*offset, values['mAP'][i], label, fontsize=15, ha='right', va='bottom')
        	elif label == 'inf':
        		ax.text(values['FPS'][i] + 5*offset, values['mAP'][i]-offset, label, fontsize=15, ha='right', va='bottom')


# Adding lines
ax.plot(fixed_tau2['SwinT-Evenly-Skip']['FPS'], fixed_tau2['SwinT-Evenly-Skip']['mAP'], linestyle='-', color='#0071E3', alpha=1.0, linewidth=2.5)
ax.plot(fixed_tau2['R50-Evenly-Skip']['FPS'], fixed_tau2['R50-Evenly-Skip']['mAP'], linestyle='-', color='#E87530', alpha=1.0, linewidth=2.5)
ax.plot(fixed_tau2['SwinT-Ours-Skip']['FPS'], fixed_tau2['SwinT-Ours-Skip']['mAP'], linestyle='-', color='#0071E3', alpha=1.0, linewidth=2.5)
ax.plot(fixed_tau2['R50-Ours-Skip']['FPS'], fixed_tau2['R50-Ours-Skip']['mAP'], linestyle='-', color='#E87530', alpha=1.0, linewidth=2.5)

# Adding labels and title
ax.set_xlabel('FPS', fontname='Times New Roman', fontsize=25)
ax.set_ylabel('mAP', fontname='Times New Roman', fontsize=25)

# Setting y-axis range
ax.set_ylim(19.5, 23)

# Adding a legend
ax.legend(fontsize=15)

# Save the figure as PDF and PNG
plt.savefig('fig_FixedTau2.pdf', format='pdf', bbox_inches='tight')
plt.savefig('fig_FixedTau2.png', format='png', bbox_inches='tight')

