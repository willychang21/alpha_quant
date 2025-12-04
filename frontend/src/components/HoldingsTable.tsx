import React, { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  flexRender,
  getFilteredRowModel,
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
import { Input } from "@/components/ui/input"
import { Search, GripVertical, ArrowUpDown, ArrowUp, ArrowDown } from 'lucide-react'

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
    backgroundColor: isPinned ? 'hsl(var(--background))' : undefined, // Ensure background is opaque
  };
};

// Draggable Header Component
const DraggableTableHeader = ({ header }: { header: Header<any, unknown> }) => {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({
      id: header.id,
    });

  const style = {
    ...getCommonPinningStyles(header.column), // Apply pinning styles
    transform: CSS.Translate.toString(transform),
    transition,
    zIndex: isDragging ? 20 : (header.column.getIsPinned() ? 10 : 0), // Higher z-index for dragging
    backgroundColor: header.column.getIsPinned() ? 'hsl(var(--card))' : undefined, // Match card bg
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

interface HoldingsTableProps {
  data: {
    matrixHoldings: any[];
    etfHeaders: any[];
  };
}

const HoldingsTable: React.FC<HoldingsTableProps> = ({ data }) => {
  const { matrixHoldings = [], etfHeaders = [] } = data || {};
  const [globalFilter, setGlobalFilter] = useState('');
  const [columnOrder, setColumnOrder] = useState<ColumnOrderState>([]);
  const [columnPinning, setColumnPinning] = useState<ColumnPinningState>({
    left: ['ticker'], // Pin 'ticker' to the left
  });

  // Define Columns
  const columns = useMemo<ColumnDef<any>[]>(() => {
    const baseCols: ColumnDef<any>[] = [
      {
        accessorKey: 'ticker',
        header: 'Ticker',
        size: 100,
        enablePinning: true, // Allow pinning
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
        size: 250,
        cell: info => <div className="truncate max-w-[200px]" title={info.getValue() as string}>{info.getValue() as string}</div>
      },
      {
        accessorKey: 'percentOfPortfolio',
        header: '% Portfolio',
        size: 150,
        cell: info => <span className="font-mono ">{(info.getValue() as number).toFixed(2)}%</span>
      },
      {
        accessorKey: 'fundsHolding',
        header: '# Funds',
        size: 150,
        cell: info => <span className="font-mono text-muted-foreground">{info.getValue() as number}</span>
      }
    ];

    const etfCols: ColumnDef<any>[] = etfHeaders.map(etf => ({
      id: etf.id, // Use ID for column ID
      accessorFn: (row: any) => row.etfs[etf.id], // Access nested data
      header: etf.label,
      size: 100,
      cell: info => {
        const val = info.getValue() as number;
        return val > 0 
          ? <span className="text-primary font-medium font-mono">{val.toFixed(2)}%</span>
          : <span className="text-muted-foreground font-mono">-</span>;
      }
    }));

    const directCol: ColumnDef<any> = {
      accessorKey: 'direct',
      header: 'Direct',
      size: 100,
      cell: info => {
        const val = info.getValue() as number;
        return val > 0
          ? <span className="text-accent font-medium font-mono">{val.toFixed(2)}%</span>
          : <span className="text-muted-foreground font-mono">-</span>;
      }
    };

    return [...baseCols, ...etfCols, directCol];
  }, [etfHeaders]);

  // Initialize column order if empty
  useMemo(() => {
    if (columnOrder.length === 0) {
      setColumnOrder(columns.map(c => c.id || (c as any).accessorKey as string));
    }
  }, [columns, columnOrder.length]);


  const table = useReactTable({
    data: matrixHoldings,
    columns,
    state: {
      globalFilter,
      columnOrder,
      columnPinning,
    },
    onGlobalFilterChange: setGlobalFilter,
    onColumnOrderChange: setColumnOrder,
    onColumnPinningChange: setColumnPinning,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    columnResizeMode: 'onChange',
    enableColumnResizing: true,
    enablePinning: true,
  });

  // DnD Sensors
  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8, // Require movement of 8px before drag starts (prevents accidental drags on click)
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
        {/* Search Bar */}
        <div className="flex flex-col sm:flex-row sm:justify-between sm:items-center gap-4 mb-6">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              type="text"
              placeholder="Search by ticker or name..."
              value={globalFilter ?? ''}
              onChange={(e) => setGlobalFilter(e.target.value)}
              className="pl-10 bg-gray-50 border-gray-200 focus:bg-white transition-smooth focus:ring-2 focus:ring-primary/50 hover:border-primary/30"
            />
          </div>
        </div>

        {/* Table */}
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
                  {table.getRowModel().rows.map((row, idx) => (
                    <motion.tr
                      key={row.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.2, delay: idx * 0.03 }}
                      className={`table-hover-row border-b transition-colors hover:bg-muted/50 data-[state=selected]:bg-muted ${idx % 2 === 0 ? 'bg-transparent' : 'bg-muted/5'}`}
                    >
                      {row.getVisibleCells().map(cell => {
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
                  ))}
                </TableBody>
              </Table>
            </DndContext>
          </div>
        </div>

        <div className="mt-4 text-sm text-muted-foreground">
          Showing {table.getRowModel().rows.length} of {matrixHoldings.length} holdings
        </div>
      </div>
    </div>
  );
};

export default HoldingsTable;
