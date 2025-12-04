import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  ColumnDef,
  ColumnOrderState,
  ColumnPinningState,
  Header,
} from '@tanstack/react-table';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  horizontalListSortingStrategy,
  useSortable,
  sortableKeyboardCoordinates,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Button } from "@/components/ui/button"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ChevronDown, LayoutGrid, Columns, TrendingUp, TrendingDown, GripVertical, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react';

// Helper for sticky styles
const getCommonPinningStyles = (column: any) => {
  const isPinned = column.getIsPinned();
  const isLastLeftPinned = isPinned === 'left' && column.getIsLastColumn('left');
  const isFirstRightPinned = isPinned === 'right' && column.getIsFirstColumn('right');

  return {
    boxShadow: isLastLeftPinned
      ? '-4px 0 4px -4px gray inset'
      : isFirstRightPinned
      ? '4px 0 4px -4px gray inset'
      : undefined,
    left: isPinned === 'left' ? `${column.getStart('left')}px` : undefined,
    right: isPinned === 'right' ? `${column.getAfter('right')}px` : undefined,
    opacity: isPinned ? 0.97 : 1,
    position: isPinned ? 'sticky' : 'relative' as const,
    width: column.getSize(),
    zIndex: isPinned ? 10 : 0,
    backgroundColor: isPinned ? 'hsl(var(--background))' : undefined,
  };
};

// Draggable Header Component (Reused logic)
const DraggableTableHeader = ({ header }: { header: Header<any, unknown> }) => {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({
      id: header.id,
    });

  const style = {
    ...getCommonPinningStyles(header.column),
    transform: CSS.Translate.toString(transform),
    transition,
    zIndex: isDragging ? 20 : (header.column.getIsPinned() ? 10 : 0),
    backgroundColor: header.column.getIsPinned() ? 'hsl(var(--card))' : undefined,
  } as React.CSSProperties;

  return (
    <TableHead
      ref={setNodeRef}
      style={style}
      className={`relative group ${isDragging ? 'bg-accent/20' : ''}`}
      colSpan={header.colSpan}
    >
      <div className="flex items-center justify-center space-x-2 h-full w-full">
        {/* Drag Handle */}
        <button
          {...attributes}
          {...listeners}
          className="cursor-move opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-muted rounded absolute left-1"
        >
          <GripVertical className="h-4 w-4 text-muted-foreground" />
        </button>
        
        {/* Header Content (Sortable) */}
        <div 
          className="flex items-center cursor-pointer select-none gap-1"
          onClick={header.column.getToggleSortingHandler()}
        >
          {flexRender(header.column.columnDef.header, header.getContext())}
          {{
            asc: <ArrowUp className="h-4 w-4 ml-1" />,
            desc: <ArrowDown className="h-4 w-4 ml-1" />,
          }[header.column.getIsSorted() as string] ?? <ArrowUpDown className="h-3 w-3 ml-1 opacity-0 group-hover:opacity-50" />}
        </div>
      </div>

      {/* Resize Handle */}
      <div
        onMouseDown={header.getResizeHandler()}
        onTouchStart={header.getResizeHandler()}
        className={`absolute right-0 top-0 h-full w-1 cursor-col-resize bg-border/50 hover:bg-primary/50 touch-none select-none ${
          header.column.getIsResizing() ? 'bg-primary w-1.5' : ''
        }`}
      />
    </TableHead>
  );
};

interface ProfitLossTableProps {
  data: {
    profitLossHoldings: any[];
  };
}

const ProfitLossTable: React.FC<ProfitLossTableProps> = ({ data }) => {
  const { profitLossHoldings = [] } = data || {};
  const [groupBy, setGroupBy] = React.useState('Ungrouped');
  const [expandedGroups, setExpandedGroups] = React.useState<Record<string, boolean>>({});
  const [open, setOpen] = React.useState(false);
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([]);
  const [columnPinning, setColumnPinning] = useState<ColumnPinningState>({
    left: ['ticker'],
  });

  // Convert country code to flag emoji
  const getFlagEmoji = (countryCode: string) => {
    const codePoints = countryCode
      .toUpperCase()
      .split('')
      .map(char => 127397 + char.charCodeAt(0));
    return String.fromCodePoint(...codePoints);
  };

  const toggleGroup = (group: string) => {
    setExpandedGroups(prev => ({ ...prev, [group]: !prev[group] }));
  };

  const handleGroupChange = (type: string) => {
    setGroupBy(type);
    if (type !== 'Ungrouped') {
      const groups: Record<string, boolean> = {};
      const uniqueGroups = [...new Set(profitLossHoldings.map(h => {
        if (type === 'Instrument Type') return h.type;
        if (type === 'Sector') return h.sector;
        if (type === 'Industry') return h.industry;
        if (type === 'Country') return h.country;
        return 'Unknown';
      }))];
      uniqueGroups.forEach(g => groups[g] = true);
      setExpandedGroups(groups);
    }
  };

  const toggleAll = (expand: boolean) => {
    const groups: Record<string, boolean> = {};
    const uniqueGroups = [...new Set(profitLossHoldings.map(h => {
      if (groupBy === 'Instrument Type') return h.type;
      if (groupBy === 'Sector') return h.sector;
      if (groupBy === 'Industry') return h.industry;
      if (groupBy === 'Country') return h.country;
      return 'Unknown';
    }))];
    uniqueGroups.forEach(g => groups[g] = expand);
    setExpandedGroups(groups);
  };

  // Group Data Logic
  const groupedData = useMemo(() => {
    if (groupBy === 'Ungrouped') return { 'All': profitLossHoldings };
    
    return profitLossHoldings.reduce((acc: Record<string, any[]>, item) => {
      let key = 'Unknown';
      if (groupBy === 'Instrument Type') key = item.type;
      else if (groupBy === 'Sector') key = item.sector;
      else if (groupBy === 'Industry') key = item.industry;
      else if (groupBy === 'Country') key = item.country;
      
      if (!acc[key]) acc[key] = [];
      acc[key].push(item);
      return acc;
    }, {});
  }, [groupBy, profitLossHoldings]);

  // Define Columns
  const columns = useMemo<ColumnDef<any>[]>(() => [
    {
      accessorKey: 'ticker',
      header: 'Ticker',
      size: 100,
      enablePinning: true,
      cell: info => (
        <Link 
          to={`/fundamental/${info.getValue()}`} 
          className="badge font-mono font-semibold hover:underline cursor-pointer text-primary hover:text-primary/80"
        >
          {info.getValue() as string}
        </Link>
      )
    },
    {
      accessorKey: 'name',
      header: 'Name',
      size: 200,
      cell: info => <div className="truncate max-w-[200px]" title={info.getValue() as string}>{info.getValue() as string}</div>
    },
    {
      accessorKey: 'priceChangePercent1D',
      header: 'Price Chg. % (1D)',
      size: 140,
      cell: info => {
        const val = info.getValue() as number;
        return (
          <div className={`font-mono font-semibold flex items-center gap-1 ${val >= 0 ? 'text-success' : 'text-danger'}`}>
             {val >= 0 ? <TrendingUp className="h-3 w-3" /> : <TrendingDown className="h-3 w-3" />}
             {val >= 0 ? '+' : ''}{val.toFixed(2)}%
          </div>
        );
      }
    },
    {
      accessorKey: 'pl1D',
      header: 'P/L (1D)',
      size: 100,
      cell: info => {
        const val = info.getValue() as number;
        return (
          <span className={`font-mono font-semibold ${val >= 0 ? 'text-success' : 'text-danger'}`}>
            ${val >= 0 ? '+' : ''}{val.toFixed(2)}
          </span>
        );
      }
    },
    {
      accessorKey: 'percentOfPortfolio',
      header: 'Weight',
      size: 100,
      cell: info => <span className="font-mono">{(info.getValue() as number).toFixed(2)}%</span>
    },
    {
      accessorKey: 'flag',
      header: 'Flag',
      size: 60,
      cell: info => <span className="text-lg">{getFlagEmoji(info.getValue() as string)}</span>
    },
    {
      accessorKey: 'purchaseDate',
      header: 'Purchase Date',
      size: 120,
      cell: info => <span className="font-mono text-xs">{info.getValue() as string}</span>
    },
    {
      accessorKey: 'quantity',
      header: 'Quantity',
      size: 100,
      cell: info => <span className="font-mono">{(info.getValue() as number).toFixed(2)}</span>
    },
    {
      accessorKey: 'averageCost',
      header: 'Avg Cost',
      size: 100,
      cell: info => <span className="font-mono">${info.getValue() as number}</span>
    },
    {
      accessorKey: 'lastPrice',
      header: 'Last Price',
      size: 100,
      cell: info => <span className="font-mono font-medium">${(info.getValue() as number).toFixed(2)}</span>
    },
    {
      accessorKey: 'marketValue',
      header: 'Market Value',
      size: 120,
      cell: info => <span className="font-mono font-semibold">${(info.getValue() as number).toFixed(2)}</span>
    },
    {
      accessorKey: 'plExclFX',
      header: 'P/L (excl. FX)',
      size: 120,
      cell: info => <span className="font-mono text-sm">{info.getValue() as number}</span>
    },
    {
      accessorKey: 'plFromFX',
      header: 'P/L (from FX)',
      size: 120,
      cell: info => <span className="font-mono text-sm">{info.getValue() as number}</span>
    },
    {
      accessorKey: 'pl',
      header: 'P/L',
      size: 100,
      cell: info => <span className="font-mono font-semibold">{info.getValue() as number}</span>
    },
    {
      accessorKey: 'plPercent',
      header: 'P/L %',
      size: 100,
      cell: info => <span className="font-mono font-semibold">{info.getValue() as number}</span>
    },
    {
      accessorKey: 'totalReturnPercent',
      header: 'Total Return %',
      size: 120,
      cell: info => <span className="font-mono font-semibold">{info.getValue() as number}</span>
    },
  ], []);

  // Initialize column order
  useMemo(() => {
    if (columnOrder.length === 0) {
      setColumnOrder(columns.map(c => (c as any).accessorKey as string));
    }
  }, [columns, columnOrder.length]);

  const table = useReactTable({
    data: profitLossHoldings,
    columns,
    state: {
      columnOrder,
      columnPinning,
    },
    onColumnOrderChange: setColumnOrder,
    onColumnPinningChange: setColumnPinning,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getRowId: row => row.ticker, // Important for mapping
    columnResizeMode: 'onChange',
    enableColumnResizing: true,
    enablePinning: true,
  });

  // Create a map of rows for quick access during manual grouping rendering
  const rowMap = useMemo(() => {
    const map: Record<string, any> = {};
    table.getRowModel().rows.forEach(row => {
      map[row.original.ticker] = row;
    });
    return map;
  }, [table.getRowModel().rows]);

  // DnD Sensors
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (active && over && active.id !== over.id) {
      setColumnOrder((order) => {
        const oldIndex = order.indexOf(active.id as string);
        const newIndex = order.indexOf(over.id as string);
        return arrayMove(order, oldIndex, newIndex);
      });
    }
  };

  return (
    <div className="space-y-4">
      <div className="glass-card p-6 rounded-2xl">
        <div className="flex flex-wrap justify-end gap-3 mb-6">
          {/* Group By Popover */}
          <Popover open={open} onOpenChange={setOpen}>
            <PopoverTrigger asChild>
              <Button 
                variant="outline" 
                size="sm"
                className={`rounded-full border-2 px-4 shadow-sm transition-all flex items-center ${
                  open
                    ? "border-primary bg-primary/90 text-primary-foreground shadow-primary/30"
                    : "border-border/80 bg-card/70 hover:border-primary/60 hover:bg-primary/10"
                }`}
              >
                <LayoutGrid className="mr-2 h-4 w-4 shrink-0" />
                <span>Group</span>
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 glass-card z-50">
              <div className="grid gap-4">
                <div className="space-y-2">
                  <h4 className="font-semibold leading-none">Grouping Options</h4>
                  <p className="text-sm text-muted-foreground">
                    Select how you want to group your holdings.
                  </p>
                </div>
                <RadioGroup value={groupBy} onValueChange={handleGroupChange} className="grid gap-2">
                  {['Ungrouped', 'Sector', 'Industry', 'Country', 'Instrument Type'].map((option) => (
                    <div key={option} className="flex items-center space-x-2 transition-smooth hover:bg-muted/50 p-2 rounded-md">
                      <RadioGroupItem value={option} id={option} />
                      <Label htmlFor={option} className="cursor-pointer flex-1">{option}</Label>
                    </div>
                  ))}
                </RadioGroup>
                <div className="border-t pt-4 flex justify-between items-center">
                  <span className="text-sm font-medium">Group Folding</span>
                  <div className="flex gap-2">
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => toggleAll(false)} 
                      disabled={groupBy === 'Ungrouped'}
                      className="h-8 text-xs transition-smooth"
                    >
                      Collapse All
                    </Button>
                    <Button 
                      variant="outline" 
                      size="sm" 
                      onClick={() => toggleAll(true)} 
                      disabled={groupBy === 'Ungrouped'}
                      className="h-8 text-xs transition-smooth"
                    >
                      Expand All
                    </Button>
                  </div>
                </div>
              </div>
            </PopoverContent>
          </Popover>
          
          {/* Columns Button (Placeholder for now, or could trigger column visibility) */}
          <Button 
            variant="outline" 
            size="sm" 
            className="rounded-full border-2 border-border/80 bg-card/70 px-4 shadow-sm transition-all hover:border-primary/60 hover:bg-primary/10 flex items-center"
          >
            <Columns className="mr-2 h-4 w-4 shrink-0" />
            <span>Columns</span>
          </Button>
        </div>

        <div className="rounded-xl border border-border/50 overflow-hidden bg-card/50">
          <div className="overflow-x-auto">
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <Table style={{ width: table.getTotalSize() }}>
                <TableHeader>
                  {table.getHeaderGroups().map(headerGroup => (
                    <TableRow key={headerGroup.id} className="sticky-header border-b-2 border-border/80 hover:bg-transparent">
                      <SortableContext
                        items={columnOrder}
                        strategy={horizontalListSortingStrategy}
                      >
                        {headerGroup.headers.map(header => (
                          <DraggableTableHeader key={header.id} header={header} />
                        ))}
                      </SortableContext>
                    </TableRow>
                  ))}
                </TableHeader>
                <TableBody>
                  {Object.entries(groupedData).map(([group, holdings]) => (
                    <React.Fragment key={group}>
                      {groupBy !== 'Ungrouped' && (
                        <TableRow 
                          className="bg-gradient-to-r from-muted/60 to-muted/30 hover:from-muted/80 hover:to-muted/50 cursor-pointer font-semibold transition-smooth border-b border-border/50" 
                          onClick={() => toggleGroup(group)}
                        >
                          <TableCell colSpan={columns.length} className="py-3">
                            <div className="flex items-center gap-2">
                              <div className={`transition-transform duration-300 ${expandedGroups[group] ? 'rotate-0' : '-rotate-90'}`}>
                                <ChevronDown className="h-5 w-5 text-primary" />
                              </div>
                              <span className="text-base">{group}</span>
                              <span className="text-xs text-muted-foreground ml-2">({holdings.length} holdings)</span>
                            </div>
                          </TableCell>
                        </TableRow>
                      )}
                      {(groupBy === 'Ungrouped' || expandedGroups[group]) && holdings.map((holding: any, idx: number) => {
                        const row = rowMap[holding.ticker];
                        if (!row) return null;
                        return (
                          <motion.tr
                            key={row.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.2, delay: idx * 0.02 }}
                            className={`table-hover-row border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted ${idx % 2 === 0 ? 'bg-transparent' : 'bg-muted/5'}`}
                          >
                            {row.getVisibleCells().map((cell: any) => {
                              const { column } = cell;
                              return (
                                <TableCell 
                                  key={cell.id} 
                                  style={{ 
                                    ...getCommonPinningStyles(column),
                                    backgroundColor: column.getIsPinned() ? (idx % 2 === 0 ? 'hsl(var(--card))' : 'hsl(var(--muted))') : undefined,
                                    zIndex: column.getIsPinned() ? 1 : 0,
                                  } as React.CSSProperties}
                                >
                                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                                </TableCell>
                              );
                            })}
                          </motion.tr>
                        );
                      })}
                    </React.Fragment>
                  ))}
                </TableBody>
              </Table>
            </DndContext>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProfitLossTable;
