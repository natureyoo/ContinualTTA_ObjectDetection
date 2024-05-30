import matplotlib.pyplot as plt

# Data
new_data = {
    'SwinT-Evenly_Skip': {'FPS': [10.77731375, 12.89698533, 15.20334474, 18.05],
                          'mAP': [22.2, 21.9, 21.2, 20.6],
                          'labels': ['1 per 2', '1 per 3', '1 per 5', '1 per 10']},
    'SwinT-Ours': {'FPS': [17.03577513, 17.05, 17.27, 17.35, 17.45, 17.58],
                   'mAP': [21.72240454, 21.7, 21.64097137, 21.59528936, 21.58, 21.57],
                   'labels': ['1.0', '1.05', '1.1', '1.15', '1.2', '-']},
    'R50-Evenly_Skip': {'FPS': [11.7, 14.23, 17.03940362, 20.79],
                        'mAP': [21.9, 21.7, 21.2, 20.36],
                        'labels': ['1 per 2', '1 per 3', '1 per 5', '1 per 10']},
    'R50-Ours': {'FPS': [20.3757764, 21.24, 21.32334531, 21.33553635, 21.38735984, 21.53075926],
                 'mAP': [21.99322769, 21.9, 21.72173945, 21.65, 21.6, 21.1],
                   'labels': ['1.0', '1.05', '1.1', '1.15', '1.2', '-']},
}
# Set the style to Apple Keynote style
plt.style.use('default')  # Reset to default style first
plt.style.use({
    'font.family': 'serif',
    'font.serif': 'Times New Roman',
    'font.size': 12
})

# Create a single scatter plot
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('Scatter Plot of FPS vs mAP (Keynote Style)')

# Plot each scatter plot with consistent colors and markers
for (name, values) in new_data.items():
    if 'SwinT' in name:
        # color = '#6F777D'  # Light gray-blue for SwinT
        color = '#0071E3'
    else:
        color = '#E87530'  # Burnt orange for R50
    
    if 'Evenly_Skip' in name:
        marker = 'x'
    else:
        marker = 's'
    
    # ax.scatter(values['FPS'], values['mAP'], label=name, color=color, marker=marker, alpha=alpha, edgecolors='none')
    ax.scatter(values['FPS'], values['mAP'], label=name, color=color, marker=marker, edgecolors='none')
    # Adding labels
    if 'Evenly_Skip' in name:
        offset = 0.1
        for i, label in enumerate(values['labels']):
            ax.text(values['FPS'][i] + offset, values['mAP'][i] + offset, label, fontsize=8, ha='right', va='bottom')
    else:
        offset = 0.2
        for i, label in enumerate(values['labels']):
            ax.text(values['FPS'][i] + offset, values['mAP'][i], label, fontsize=8, ha='right', va='bottom')

# Adding lines
ax.plot(new_data['SwinT-Evenly_Skip']['FPS'], new_data['SwinT-Evenly_Skip']['mAP'], linestyle='-', color='#0071E3', alpha=1.0, linewidth=1.5)
ax.plot(new_data['R50-Evenly_Skip']['FPS'], new_data['R50-Evenly_Skip']['mAP'], linestyle='-', color='#E87530', alpha=1.0, linewidth=1.5)
ax.plot(new_data['SwinT-Ours']['FPS'], new_data['SwinT-Ours']['mAP'], linestyle='-', color='#0071E3', alpha=1.0, linewidth=1.5)
ax.plot(new_data['R50-Ours']['FPS'], new_data['R50-Ours']['mAP'], linestyle='-', color='#E87530', alpha=1.0, linewidth=1.5)

# Adding labels and title
ax.set_xlabel('FPS')
ax.set_ylabel('mAP')

# Setting wider y-axis range
ax.set_ylim(20, 23)

# Adding a legend
ax.legend()

# Save the figure as PDF and PNG
plt.savefig('combined_scatter_plot_keynote_style_new_data.pdf', format='pdf', bbox_inches='tight')
plt.savefig('combined_scatter_plot_keynote_style_new_data.png', format='png', bbox_inches='tight')

# Display the plot
plt.show()
